## B known, Find A
function identify_model(prob::IdentifyProblemX, ::Val{L∞Norm}, ::Val{BKnown},
        q_seq, μA_list, δA_list, B_list, solver)
    D = state_dim(prob)
    n_obs = prob.n_obs
    n_mode = prob.n_mode
    u_seq = prob.u_seq
    x1_seq = prob.x1_seq
    x2_seq = prob.x2_seq

    model = Model(solver)
    A_list = [@variable(model, [1:D, 1:D], base_name=string("A", q))
        for q = 1:n_mode]
    r = @variable(model, base_name="r")

    for q = 1:n_mode
        A = A_list[q]
        δ = δA_list[q]
        if !isinf(δ)
            @constraint(model, A - μA_list[q] .≤ +δ)
            @constraint(model, A - μA_list[q] .≥ -δ)
        end
        B = B_list[q]
        for (k, i) in enumerate(filter(i -> q_seq[i] == q, 1:n_obs))
            x1 = x1_seq[i]
            u = u_seq[i]
            x2 = x2_seq[i] - B*u
            @constraint(model, A*x1 - x2 .≤ +r)
            @constraint(model, A*x1 - x2 .≥ -r)
        end
    end

    @objective(model, Min, r)

    optimize!(model)

    Aopt_list = [zeros(D, D) for q = 1:n_mode]
    ropt = Inf
    flag = false 

    if isone(Int(primal_status(model)))
        flag = true
        ropt = Float64(value(r))
        for q = 1:n_mode
            Aopt_list[q] = value.(A_list[q])
        end
    end

    return Aopt_list, ropt, flag
end

## A known, Find B
function identify_model(prob::IdentifyProblemX, ::Val{L∞Norm}, ::Val{AKnown},
        q_seq, A_list, μB_list, δB_list, solver)
    D = state_dim(prob)
    E = input_dim(prob)
    n_obs = prob.n_obs
    n_mode = prob.n_mode
    u_seq = prob.u_seq
    x1_seq = prob.x1_seq
    x2_seq = prob.x2_seq

    model = Model(solver)
    B_list = [@variable(model, [1:D, 1:E], base_name=string("B", q))
        for q = 1:n_mode]
    r = @variable(model, base_name="r")

    for q = 1:n_mode
        B = B_list[q]
        δ = δB_list[q]
        if !isinf(δ)
            @constraint(model, B - μB_list[q] .≤ +δ)
            @constraint(model, B - μB_list[q] .≥ -δ)
        end
        A = A_list[q]
        for (k, i) in enumerate(filter(i -> q_seq[i] == q, 1:n_obs))
            δx = x2_seq[i] - A*x1_seq[i]
            u = u_seq[i]
            @constraint(model, B*u - δx .≤ +r)
            @constraint(model, B*u - δx .≥ -r)
        end
    end

    @objective(model, Min, r)

    optimize!(model)

    Bopt_list = [zeros(D, E) for q = 1:n_mode]
    ropt = Inf
    flag = false 

    if isone(Int(primal_status(model)))
        flag = true
        ropt = Float64(value(r))
        for q = 1:n_mode
            Bopt_list[q] = value.(B_list[q])
        end
    end

    return Bopt_list, ropt, flag
end

# Find A B
function identify_model(prob::IdentifyProblemX, ::Val{L∞Norm}, ::Nothing,
        q_seq, μA_list, δA_list, μB_list, δB_list, solver)
    D = state_dim(prob)
    E = input_dim(prob)
    n_obs = prob.n_obs
    n_mode = prob.n_mode
    u_seq = prob.u_seq
    x1_seq = prob.x1_seq
    x2_seq = prob.x2_seq

    model = Model(solver)
    A_list = [@variable(model, [1:D, 1:D], base_name=string("A", q))
        for q = 1:n_mode]
    B_list = [@variable(model, [1:D, 1:E], base_name=string("B", q))
        for q = 1:n_mode]
    r = @variable(model, base_name="r")

    for q = 1:n_mode
        A = A_list[q]
        δ = δA_list[q]
        if !isinf(δ)
            @constraint(model, A - μA_list[q] .≤ +δ)
            @constraint(model, A - μA_list[q] .≥ -δ)
        end
        B = B_list[q]
        δ = δA_list[q]
        if !isinf(δ)
            @constraint(model, B - μB_list[q] .≤ +δ)
            @constraint(model, B - μB_list[q] .≥ -δ)
        end
        for (k, i) in enumerate(filter(i -> q_seq[i] == q, 1:n_obs))
            x1 = x1_seq[i]
            u = u_seq[i]
            x2 = x2_seq[i]
            @constraint(model, A*x1 + B*u - x2 .≤ +r)
            @constraint(model, A*x1 + B*u - x2 .≥ -r)
        end
    end

    @objective(model, Min, r)

    optimize!(model)

    Aopt_list = [zeros(D, D) for q = 1:n_mode]
    Bopt_list = [zeros(D, E) for q = 1:n_mode]
    ropt = Inf
    flag = false 

    if isone(Int(primal_status(model)))
        flag = true
        ropt = Float64(value(r))
        for q = 1:n_mode
            Aopt_list[q] = value.(A_list[q])
            Bopt_list[q] = value.(B_list[q])
        end
    end

    return Aopt_list, Bopt_list, ropt, flag
end