function identify_disc_model(
        prob::IdentifyProblemX, ::Val{L∞Norm}, ::Val{BKnown},
        ::Val{PriorIdentical}, μA, δA, B, ϵ, solver;
        print_flag=true, Dbin=1000)
    D = state_dim(prob)
    n_obs = prob.n_obs
    n_mode = prob.n_mode
    u_seq = prob.u_seq
    x1_seq = prob.x1_seq
    x2_seq = prob.x2_seq

    As_list = [μA for q = 1:n_mode] # strict
    qs_seq = zeros(Int, n_obs)

    Alb = μA .- δA
    Aub = μA .+ δA

    a_seq = Vector{Vector{Float64}}(undef, n_obs)
    b_seq = Vector{Vector{Float64}}(undef, n_obs)
    
    for i = 1:n_obs
        a_seq[i] = x1_seq[i]
        b_seq[i] = x2_seq[i] - B*u_seq[i]
    end

    i_coll = collect(1:n_obs)
    a_max = maximum(x -> norm(x, Inf), a_seq)
    b_max = maximum(x -> norm(x, Inf), b_seq)
    Cbin = Dbin*(a_max + b_max)
    flag = false
    q = 1

    while !isempty(i_coll) && q ≤ n_mode
        X, io_coll, flag = _max_FS(a_seq, b_seq, i_coll,
            Alb, Aub, ϵ, solver, Cbin)
        if print_flag
            @printf("q (flag: %s): %d, n active: %d\n",
                flag, q, length(io_coll))
        end
        (!flag || isempty(io_coll)) && break
        As_list[q] = X
        for i in io_coll
            qs_seq[i] = q
        end
        setdiff!(i_coll, io_coll)
        q += 1
    end

    @printf("\nTerminated (flag: %s): i rest: %d, q rest: %d\n",
        flag, length(i_coll), n_mode - q + 1)

    return qs_seq, As_list, i_coll, flag
end

function _max_FS(a_seq, b_seq, i_coll, Xlb, Xub, ϵ, solver, Cbin)
    n_eq = length(i_coll)
    E = length(a_seq[1])
    D = length(b_seq[1])
    
    model = Model(solver)
    X = @variable(model, [1:D, 1:E], base_name=string("X"))
    bin_seq = [@variable(model, base_name=string("bin_seq", i),
        binary=true) for i = 1:n_eq]

    for (k, i) in enumerate(i_coll)
        a = a_seq[i]
        b = b_seq[i]
        bin = bin_seq[k]
        @constraint(model, X*a - b .≤ +ϵ + Cbin*(1 - bin))
        @constraint(model, X*a - b .≥ -ϵ - Cbin*(1 - bin))
    end

    @constraint(model, X .≥ Xlb)
    @constraint(model, X .≤ Xub)

    @objective(model, Max, sum(bin_seq))

    optimize!(model)

    Xopt = zeros(D, E)
    iopt_coll = Int[]
    flag = false

    if isone(Int(primal_status(model)))
        flag = true
        Xopt = value.(X)
        binopt_seq = value.(bin_seq)
        for k = 1:n_eq
            if binopt_seq[k] > 0.5
                push!(iopt_coll, i_coll[k])
            end
        end
    end
    
    return Xopt, iopt_coll, flag
end