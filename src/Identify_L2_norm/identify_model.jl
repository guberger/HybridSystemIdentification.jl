## B known, Find A
function identify_model(prob::IdentifyProblemX, ::Val{L2Norm}, ::Val{BKnown},
        q_seq, μA_list, σA_list, B_list)
    D = state_dim(prob)
    n_obs = prob.n_obs
    n_mode = prob.n_mode
    u_seq = prob.u_seq
    x1_seq = prob.x1_seq
    x2_seq = prob.x2_seq

    A_list = Vector{Matrix{Float64}}(undef, n_mode)

    for q = 1:n_mode
        n_q = sum(q_seq .== q)
        B = B_list[q]
        M = Matrix{Float64}(undef, D + n_q, D)
        N = zeros(D + n_q, D)
        M[1:D, 1:D] = Matrix(I, D, D) / (σA_list[q]^2)
        N[1:D, 1:D] = μA_list[q]'
        for (k, i) in enumerate(filter(i -> q_seq[i] == q, 1:n_obs))
            x1 = x1_seq[i]
            u = u_seq[i]
            x2 = x2_seq[i] - B*u
            M[D + k, :] = x1'
            N[D + k, :] = x2'
        end
        A_list[q] = (M \ N)'
    end

    return A_list
end

## A known, Find B
function identify_model(prob::IdentifyProblemX, ::Val{L2Norm}, ::Val{AKnown},
            q_seq, A_list, μB_list, σB_list)
    D = state_dim(prob)
    E = input_dim(prob)
    n_obs = prob.n_obs
    n_mode = prob.n_mode
    u_seq = prob.u_seq
    x1_seq = prob.x1_seq
    x2_seq = prob.x2_seq

    B_list = Vector{Matrix{Float64}}(undef, n_mode)

    for q = 1:n_mode
        n_q = sum(q_seq .== q)
        A = A_list[q]
        M = Matrix{Float64}(undef, E + n_q, E)
        N = zeros(E + n_q, D)
        M[1:E, 1:E] = Matrix(I, E, E) / (σB_list[q]^2)
        N[1:E, 1:D] = μB_list[q]'
        for (k, i) in enumerate(filter(i -> q_seq[i] == q, 1:n_obs))
            δx = x2_seq[i] - A*x1_seq[i]
            u = u_seq[i]
            M[E + k, :] = u'
            N[E + k, :] = δx'
        end
        B_list[q] = (M \ N)'
    end

    return B_list
end

# Find A B
function identify_model(prob::IdentifyProblemX, ::Val{L2Norm}, ::Nothing,
        q_seq, μA_list, σA_list, μB_list, σB_list)
    D = state_dim(prob)
    E = input_dim(prob)
    n_obs = prob.n_obs
    n_mode = prob.n_mode
    u_seq = prob.u_seq
    x1_seq = prob.x1_seq
    x2_seq = prob.x2_seq

    A_list = Vector{Matrix{Float64}}(undef, n_mode)
    B_list = Vector{Matrix{Float64}}(undef, n_mode)

    for q = 1:n_mode
        n_q = sum(q_seq .== q)
        M = zeros(D + E + n_q, D + E)
        N = zeros(D + E + n_q, D)
        M[1:D, 1:D] = Matrix(I, D, D) / (σA_list[q]^2)
        N[1:D, 1:D] = μA_list[q]'
        M[D+1:D+E, D+1:D+E] = Matrix(I, E, E) / (σB_list[q]^2)
        N[D+1:D+E, 1:D] = μB_list[q]'
        for (k, i) in enumerate(filter(i -> q_seq[i] == q, 1:n_obs))
            x1 = x1_seq[i]
            u = u_seq[i]
            x2 = x2_seq[i]
            M[D + E + k, 1:D] = x1'
            M[D + E + k, D+1:D+E] = u'
            N[D + E + k, :] = x2'
        end
        AB = (M \ N)'
        A_list[q] = AB[:, 1:D]
        B_list[q] = AB[:, D+1:D+E]
    end

    return A_list, B_list
end