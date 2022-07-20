function identify_disc_model(
        prob::IdentifyProblemX, ::Val{L2Norm}, ::Val{BKnown},
        ::Val{PriorIdentical}, μA, σA, B, f_mode;
        print_period=1)
    μA_list = [μA for q = 1:prob.n_mode]
    σA_list = [σA for q = 1:prob.n_mode]
    B_list = [B for q = 1:prob.n_mode]
    f_mode_2 = isnothing(f_mode) ? (dist, q1, q2) -> (0.0, 0.0) : f_mode
    return _identify_disc_model_L2Norm(prob, Val(BKnown), true,
        μA_list, σA_list, B_list, f_mode_2, print_period)
end

function identify_disc_model(
        prob::IdentifyProblemX, ::Val{L2Norm}, ::Val{BKnown},
        ::Val{PriorDifferent}, μA_list, σA_list, B_list, f_mode;
        print_period=1)
    f_mode_2 = isnothing(f_mode) ? (dist, q1, q2) -> (0.0, 0.0) : f_mode
    return _identify_disc_model_L2Norm(prob, Val(BKnown), false,
        μA_list, σA_list, B_list, f_mode_2, print_period)
end

function identify_disc_model(
        prob::IdentifyProblemX, ::Val{L2Norm}, ::Val{AKnown},
        ::Val{PriorIdentical}, A, μB, σB, f_mode;
        print_period=1)
    A_list = [A for q = 1:prob.n_mode]
    μB_list = [μB for q = 1:prob.n_mode]
    σB_list = [σB for q = 1:prob.n_mode]
    f_mode_2 = isnothing(f_mode) ? (dist, q1, q2) -> (0.0, 0.0) : f_mode
    return _identify_disc_model_L2Norm(prob, Val(AKnown), true,
        A_list, μB_list, σB_list, f_mode_2, print_period)
end

function identify_disc_model(
        prob::IdentifyProblemX, ::Val{L2Norm}, ::Val{AKnown},
        ::Val{PriorDifferent}, A_list, μB_list, σB_list, f_mode;
        print_period=1)
    f_mode_2 = isnothing(f_mode) ? (dist, q1, q2) -> (0.0, 0.0) : f_mode
    return _identify_disc_model_L2Norm(prob, Val(AKnown), false,
        A_list, μB_list, σB_list, f_mode_2, print_period)
end

const NodeTree = Tuple{String,Int,Int,Int,Float64}

function _identify_disc_model_L2Norm(
        prob::IdentifyProblemX, ::Val{BKnown}, PI,
        μA_list, σA_list, B_list, f_mode, print_period)
    D = state_dim(prob)
    n_obs = prob.n_obs
    n_mode = prob.n_mode
    u_seq = prob.u_seq
    x1_seq = prob.x1_seq
    x2_seq = prob.x2_seq

    M_dict = Dict{NodeTree,Vector{Matrix{Float64}}}()
    pq = PriorityQueue{NodeTree,Float64}()

    ϵ = 0.0
    depth = 0
    width = PI ? 0 : n_mode
    last_mode = 0
    node = ("", last_mode, depth, width, ϵ)
    M_dict[node] = Vector{Matrix{Float64}}(undef, n_mode)
    for q = 1:n_mode
        ρ = 1/(σA_list[q]^2)
        M_dict[node][q] = [ρ*Matrix(I, D, D) μA_list[q]'; zeros(1, 2*D)]
    end

    iter = 0
    depth_max = 0
    width_max = 0

    while depth < n_obs
        q_max = min(width + 1, n_mode)
        for q = 1:q_max
            B = B_list[q]
            x1 = x1_seq[depth + 1]
            u = u_seq[depth + 1]
            x2 = x2_seq[depth + 1] - B*u
            M_list = M_dict[node]
            M_list_new = copy(M_list)
            M = M_list[q]
            M_new = [x1' x2'; M[1:D, :]]
            _triangularize_hessenberg!(M_new)
            width_new = max(width, q)
            node_new = (string(node[1], "-", q), q, depth + 1, width_new)
            cost_curr, cost_next = f_mode(n_obs - depth - 1, last_mode, q)
            ϵ_new = ϵ + sum(i -> M_new[D + 1, D + i]^2, 1:D) + cost_curr
            node_new = (string(node[1], "-", q), q, depth + 1, width_new, ϵ_new)
            value = ϵ_new + cost_next
            M_list_new[q] = M_new
            enqueue!(pq, node_new, value)
            M_dict[node_new] = M_list_new
        end
        depth_max = max(depth_max, depth + 1)
        width_max = max(width_max, q_max)
        iter += 1
        node, value = dequeue_pair!(pq)
        last_mode, depth, width, ϵ = node[2], node[3], node[4], node[5]
        if !isnothing(print_period) && mod(iter - 1, print_period) == 0
            @printf("iter: %d, depth max: %d, width_max: %d, value: %f\n",
                iter, depth_max, width_max, value)
        end
    end

    q_seq = _decode_string(node[1])
    A_list = identify_model(prob, Val(L2Norm), Val(BKnown),
        q_seq, μA_list, σA_list, B_list)

    return q_seq, A_list
end

function _identify_disc_model_L2Norm(
        prob::IdentifyProblemX, ::Val{AKnown}, PI,
        A_list, μB_list, σB_list, f_mode, print_period)
    D = state_dim(prob)
    E = input_dim(prob)
    n_obs = prob.n_obs
    n_mode = prob.n_mode
    u_seq = prob.u_seq
    x1_seq = prob.x1_seq
    x2_seq = prob.x2_seq

    M_dict = Dict{NodeTree,Vector{Matrix{Float64}}}()
    pq = PriorityQueue{NodeTree,Float64}()

    ϵ = 0.0
    depth = 0
    width = PI ? 0 : n_mode
    last_mode = 0
    node = ("", last_mode, depth, width, ϵ)
    M_dict[node] = Vector{Matrix{Float64}}(undef, n_mode)
    for q = 1:n_mode
        ρ = 1/(σB_list[q]^2)
        M_dict[node][q] = [ρ*Matrix(I, E, E) μB_list[q]'; zeros(1, E + D)]
    end

    iter = 0
    depth_max = 0
    width_max = 0

    while depth < n_obs
        q_max = min(width + 1, n_mode)
        for q = 1:q_max
            A = A_list[q]
            δx = x2_seq[depth + 1] - A*x1_seq[depth + 1]
            u = u_seq[depth + 1]
            M_list = M_dict[node]
            M_list_new = copy(M_list)
            M = M_list[q]
            M_new = [u' δx'; M[1:E, :]]
            _triangularize_hessenberg!(M_new)
            width_new = max(width, q)
            cost_curr, cost_next = f_mode(n_obs - depth - 1, last_mode, q)
            ϵ_new = ϵ + sum(i -> M_new[E + 1, E + i]^2, 1:D) + cost_curr
            node_new = (string(node[1], "-", q), q, depth + 1, width_new, ϵ_new)
            value = ϵ_new + cost_next
            M_list_new[q] = M_new
            enqueue!(pq, node_new, value)
            M_dict[node_new] = M_list_new
        end
        depth_max = max(depth_max, depth + 1)
        width_max = max(width_max, q_max)
        iter += 1
        node, value = dequeue_pair!(pq)
        last_mode, depth, width, ϵ = node[2], node[3], node[4], node[5]
        if !isnothing(print_period) && mod(iter - 1, print_period) == 0
            @printf("iter: %d, depth max: %d, width_max: %d, value: %f\n",
                iter, depth_max, width_max, value)
        end
    end

    q_seq = _decode_string(node[1])
    B_list = identify_model(prob, Val(L2Norm), Val(AKnown),
        q_seq, A_list, μB_list, σB_list)

    return q_seq, B_list
end

function _decode_string(s)
    qs_seq = split(s, "-")
    q_seq = Vector{Int}(undef, length(qs_seq) - 1)
    for i = 2:length(qs_seq)
        q_seq[i - 1] = parse(Int, qs_seq[i])
    end
    return q_seq
end

# Triangularize the matrix H in place, by left-multiplying by orthogonal
# matrices. H is assumed to be upper Hessemberg.
function _triangularize_hessenberg!(H::Matrix)
    m, n = size(H)
    @inbounds for k = 1:m-1
        # Givens transformation: c*u1 + s*u2 = ρ; -s*u1 + c*u2 = 0
        c, s, ρ = LinearAlgebra.givensAlgorithm(H[k, k], H[k + 1, k])
        H[k, k] = ρ
        for j = k+1:n
            t = -conj(s)*H[k, j]
            H[k, j] = c*H[k, j] + s*H[k + 1, j]
            H[k + 1, j] = t + c*H[k + 1, j]
        end
    end
    return H
end