function identify_state_model(prob::IdentifyProblemY, ::Val{BKnown},
        q_seq, μA_list, σA_list, B_list, σd, σn,
        x0_seq, A0_list; print_period=1, max_iter=1_000_000, tol_diff=1e-6)
    D = state_dim(prob)
    n_obs = prob.n_obs
    n_mode = prob.n_mode
    u_seq = prob.u_seq
    y_seq = prob.y_seq
    ρn = 1/(σn^2)

    Avar_list = Vector{Matrix{Float64}}(undef, n_mode)
    xvar_seq = Vector{Vector{Float64}}(undef, n_obs)
    Avar_new_list = Vector{Matrix{Float64}}(undef, n_mode)
    xvar_new_seq = Vector{Vector{Float64}}(undef, n_obs)
    dAvar_list = Vector{Matrix{Float64}}(undef, n_mode)
    dxvar_seq = Vector{Vector{Float64}}(undef, n_obs)
    δAvar_list = Vector{Matrix{Float64}}(undef, n_mode)
    δxvar_seq = Vector{Vector{Float64}}(undef, n_obs)
    for q = 1:n_mode
        Avar_list[q] = copy(A0_list[q])
        Avar_new_list[q] = Matrix{Float64}(undef, D, D)
        dAvar_list[q] = Matrix{Float64}(undef, D, D)
        δAvar_list[q] = Matrix{Float64}(undef, D, D)
    end
    for i = 1:n_obs
        xvar_seq[i] = copy(x0_seq[i])
        xvar_new_seq[i] = Vector{Float64}(undef, D)
        dxvar_seq[i] = Vector{Float64}(undef, D)
        δxvar_seq[i] = Vector{Float64}(undef, D)
    end
    i_coll_list = Vector{Vector{Int}}(undef, n_mode)
    for q = 1:n_mode
        i_coll_list[q] = findall(q_seq[1:n_obs-1] .== q)
    end

    acc_A = 0.0; acc_x = 0.0; Fval = 0.0; Fval_new = 0.0; Fval_approx = 0.0
    δ = Vector{Float64}(undef, D)
    At_tmp = Matrix{Float64}(undef, D, D)
    xt_tmp = Matrix{Float64}(undef, 1, D)
    Fval = _compute_F!(Avar_list, B_list, xvar_seq, u_seq, y_seq,
        μA_list, σA_list, σd, σn, i_coll_list, n_obs, D, δ)
    Lip_const = 1.0
    iter = 0
    new_iter = true

    while iter < max_iter
        # compute gradient
        if new_iter
            _compute_dF!(Avar_list, B_list, xvar_seq, u_seq, y_seq,
                μA_list, σA_list, σd, σn, i_coll_list, n_obs,
                dAvar_list, dxvar_seq, δ, At_tmp, xt_tmp)
        end
        new_iter = false

        # compute step with given L
        # compute quadratic approx at tentative new iterate
        acc_A = 0
        Fval_approx = Fval
        for q = 1:n_mode
            ρA = 1/(σA_list[q]^2)
            mul!(δAvar_list[q], -1/(2*ρA + Lip_const), dAvar_list[q])
            norm_δA = norm(δAvar_list[q])^2
            acc_A += norm_δA
            Fval_approx += dot(dAvar_list[q], δAvar_list[q])
            Fval_approx += (ρA + Lip_const/2)*norm_δA
        end
        acc_x = 0
        for i = 1:n_obs
            mul!(δxvar_seq[i], -1/(2*ρn + Lip_const), dxvar_seq[i])
            norm_δx = norm(δxvar_seq[i])^2
            acc_x += norm_δx
            Fval_approx += dot(dxvar_seq[i], δxvar_seq[i])
            Fval_approx += (ρn + Lip_const/2)*norm_δx
        end

        # exit if step size < tol
        if sqrt(acc_A + acc_x) < tol_diff
            break
        end

        # compute tentative new iterate
        for q = 1:n_mode
            copyto!(Avar_new_list[q], Avar_list[q])
            mul!(Avar_new_list[q], 1, δAvar_list[q], 1, 1)
        end
        for i = 1:n_obs
            copyto!(xvar_new_seq[i], xvar_seq[i])
            mul!(xvar_new_seq[i], 1, δxvar_seq[i], 1, 1)
        end

        # compute value at tentative new iterate
        Fval_new = _compute_F!(
            Avar_new_list, B_list, xvar_new_seq, u_seq, y_seq,
            μA_list, σA_list, σd, σn, i_coll_list, n_obs, D, δ)

        # if quadratic approx < value at tentative new iterate, then update L
        # and compute a new tentative new iterate; otherwise, take tentative new
        # iterate as new iterate and update L 
        if Fval_new > Fval_approx
            Lip_const = 2*Lip_const
        else
            Fval = Fval_new
            for q = 1:n_mode
                copyto!(Avar_list[q], Avar_new_list[q])
            end
            for i = 1:n_obs
                copyto!(xvar_seq[i], xvar_new_seq[i])
            end
            iter += 1
            new_iter = true
            if mod(iter - 1, print_period) == 0
                @printf("iter: %d, norm(dx): %f (L: %f)\n",
                        iter, sqrt(acc_A + acc_x), Lip_const)
            end
            Lip_const = Lip_const/2
        end
    end

    @printf("Finished -> iter: %d, norm(dx): %f (L: %f)\n",
            iter, sqrt(acc_A + acc_x), Lip_const)

    return Avar_list, xvar_seq
end

function _compute_F!(A_list, B_list, x_seq, u_seq, y_seq,
        μA_list, σA_list, σd, σn, i_coll_list, n_obs, D, δ)
    acc = 0.0
    ρn = 1/(σn^2)
    ρd = 1/(σd^2)
    for i = 1:n_obs
        copyto!(δ, y_seq[i])
        mul!(δ, 1, x_seq[i], 1, -1)
        acc += ρn*norm(δ)^2
    end
    for (q, i_coll) in enumerate(i_coll_list)
        ρA = 1/(σA_list[q]^2)
        acc += ρA*sum(i -> (A_list[q][i] - μA_list[q][i])^2, D*D)
        for i in i_coll
            copyto!(δ, x_seq[i + 1])
            mul!(δ, B_list[q], u_seq[i], 1, -1)
            mul!(δ, A_list[q], x_seq[i], 1, 1)
            acc += ρd*norm(δ)^2
        end
    end
    return acc
end

function _compute_dF!(A_list, B_list, x_seq, u_seq, y_seq,
        μA_list, σA_list, σd, σn, i_coll_list, n_obs,
        dA_list, dx_seq, δ, At_tmp, xt_tmp)
    ρn = 1/(σn^2)
    ρd = 1/(σd^2)
    for i = 1:n_obs
        copyto!(dx_seq[i], y_seq[i])
        mul!(dx_seq[i], 1, x_seq[i], 2*ρn, -2*ρn) # dx_i = 2*ρn*(xk_i - y_i)
    end
    for (q, i_coll) in enumerate(i_coll_list)
        ρA = 1/(σA_list[q]^2)
        transpose!(At_tmp, A_list[q])
        copyto!(dA_list[q], A_list[q])
        mul!(dA_list[q], 1, μA_list[q], -2*ρA, 2*ρA) # dA_q = 2*ρA*(Ak_q - μA_q)
        for i in i_coll
            copyto!(δ, x_seq[i + 1])
            mul!(δ, B_list[q], u_seq[i], 1, -1)
            mul!(δ, A_list[q], x_seq[i], 1, 1) # δ = Ak_q*xk_i + B_q*u_i - xk_{i+1}
            transpose!(xt_tmp, x_seq[i])
            mul!(dA_list[q], δ, xt_tmp, 2*ρd, 1) # dA_q += 2*ρd*δ*xk_i'
            mul!(dx_seq[i], At_tmp, δ, 2*ρd, 1) # dx_i += 2*ρd*Ak_q'*δ
            mul!(dx_seq[i + 1], 1, δ, -2*ρd, 1) # dx_{i+1} += 2*ρd*δ
        end
    end
end