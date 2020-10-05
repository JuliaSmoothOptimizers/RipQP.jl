function compute_α_dual(v, dir_v)
    n = length(v)
    T = eltype(v)
    if n == 0
        return one(T)
    end
    α = one(T)
    @inbounds @simd for i=1:n
        if dir_v[i] < zero(T)
            α_new = -v[i] * T(0.999) / dir_v[i]
            if α_new < α
                α = α_new
            end
        end
    end
    return α
end

function compute_α_primal(v, dir_v, lvar, uvar)
    n = length(v)
    T = eltype(v)
    α_l, α_u = one(T), one(T)
    @inbounds @simd for i=1:n
        if dir_v[i] > zero(T)
            α_u_new = (uvar[i] - v[i]) * T(0.999) / dir_v[i]
            if α_u_new < α_u
                α_u = α_u_new
            end
        elseif dir_v[i] < zero(T)
            α_l_new = (lvar[i] - v[i]) * T(0.999) / dir_v[i]
            if α_l_new < α_l
                α_l = α_l_new
            end
        end
    end
    return min(α_l, α_u)
end

function compute_μ(x_m_lvar, uvar_m_x, s_l, s_u, nb_low, nb_upp)
    return (s_l' * x_m_lvar + s_u' * uvar_m_x) / (nb_low + nb_upp)
end

function solve_augmented_system_aff!(J_fact, Δ_aff, Δ_xλ, rc, rb, x_m_lvar, uvar_m_x,
                                     s_l, s_u, ilow, iupp,  n_cols, n_rows, n_low)

    Δ_xλ[1:n_cols] .= .-rc
    Δ_xλ[n_cols+1:end] .= .-rb
    Δ_xλ[ilow] += @views s_l[ilow]
    Δ_xλ[iupp] -= @views s_u[iupp]

    Δ_xλ = ldiv!(J_fact, Δ_xλ)
    Δ_aff[1:n_cols+n_rows] = Δ_xλ
    Δ_aff[n_cols+n_rows+1:n_cols+n_rows+n_low] = @views -s_l[ilow] - s_l[ilow].*Δ_xλ[1:n_cols][ilow]./x_m_lvar
    Δ_aff[n_cols+n_rows+n_low+1:end] = @views -s_u[iupp] + s_u[iupp].*Δ_xλ[1:n_cols][iupp]./uvar_m_x
    return Δ_aff
end

function solve_augmented_system_cc!(J_fact, Δ_cc, Δ_xλ ,Δ_aff, σ, μ, x_m_lvar, uvar_m_x,
                                    rxs_l, rxs_u, s_l, s_u, ilow, iupp, n_cols, n_rows, n_low)

    rxs_l .= @views (-σ*μ .+ Δ_aff[1:n_cols][ilow].*Δ_aff[n_rows+n_cols+1: n_rows+n_cols+n_low])
    rxs_u .= @views σ*μ .+ Δ_aff[1:n_cols][iupp].*Δ_aff[n_rows+n_cols+n_low+1: end]
    Δ_xλ .= zero(eltype(Δ_xλ))
    Δ_xλ[ilow] .+= rxs_l./x_m_lvar
    Δ_xλ[iupp] .+= rxs_u./uvar_m_x

    Δ_xλ = ldiv!(J_fact, Δ_xλ)
    Δ_cc[1:n_cols+n_rows] = Δ_xλ
    Δ_cc[n_cols+n_rows+1:n_cols+n_rows+n_low] .= @views .-(rxs_l.+s_l[ilow].*Δ_xλ[1:n_cols][ilow])./x_m_lvar
    Δ_cc[n_cols+n_rows+n_low+1:end] .= @views (rxs_u.+s_u[iupp].*Δ_xλ[1:n_cols][iupp])./uvar_m_x
    return Δ_cc
end

function iter_mehrotraPC!(x, λ, s_l, s_u, x_m_lvar, uvar_m_x, ilow, iupp, n_rows, n_cols,n_low, n_upp,
                          FloatData, IntData, rc, rb, rcNorm, rbNorm, tol_rb, tol_rc,Qx, ATλ, Ax,
                          xTQx_2, cTx, pri_obj, dual_obj, pdd, l_pdd, mean_pdd, n_Δx, small_Δx, small_μ,
                          Δt, tired, optimal, μ, k, ρ, δ, ρ_min, δ_min, J_augm, J_fact, J_P, diagind_J,
                          diag_Q, tmp_diag, Δ_aff, Δ_cc, Δ, Δ_xλ, s_l_αΔ_aff, s_u_αΔ_aff,
                          x_m_l_αΔ_aff, u_m_x_αΔ_aff, rxs_l, rxs_u, max_iter, ϵ_pdd, ϵ_μ, ϵ_rc, ϵ_rb, tol_Δx,
                          start_time, max_time, c_catch, c_pdd, display)
    T = eltype(x)

    while k<max_iter && !optimal && !tired # && !small_μ && !small_μ

            # Affine scaling direction
        tmp_diag .= -ρ
        tmp_diag[ilow] .-= @views s_l[ilow] ./ x_m_lvar
        tmp_diag[iupp] .-= @views s_u[iupp] ./ uvar_m_x
        J_augm.nzval[view(diagind_J,1:n_cols)] .= @views tmp_diag .- diag_Q
        J_augm.nzval[view(diagind_J, n_cols+1:n_rows+n_cols)] .= δ

        J_fact = try ldl_factorize!(Symmetric(J_augm, :U), J_P)
        catch
            if T == Float32
                break
                break
            end
            if c_pdd == 0 && c_catch==0
                δ *= T(1e2)
                δ_min *= T(1e2)
                ρ *= T(1e5)
                ρ_min *= T(1e5)
            elseif c_pdd == 0 && c_catch != 0
                δ *= T(1e1)
                δ_min *= T(1e1)
                ρ *= T(1e0)
                ρ_min *= T(1e0)
            elseif c_pdd != 0 && c_catch==0
                δ *= T(1e5)
                δ_min *= T(1e5)
                ρ *= T(1e5)
                ρ_min *= T(1e5)
            else
                δ *= T(1e1)
                δ_min *= T(1e1)
                ρ *= T(1e1)
                ρ_min *= T(1e1)
            end
            c_catch += 1
            tmp_diag .= -ρ
            tmp_diag[ilow] .-= @views s_l[ilow] ./ x_m_lvar
            tmp_diag[iupp] .-= @views s_u[iupp] ./ uvar_m_x
            J_augm.nzval[view(diagind_J,1:n_cols)] .= @views tmp_diag .- diag_Q
            J_augm.nzval[view(diagind_J, n_cols+1:n_rows+n_cols)] .= δ
            J_fact = ldl_factorize!(Symmetric(J_augm, :U), J_P)
        end

        if c_catch >= 4
            break
        end

        Δ_aff = solve_augmented_system_aff!(J_fact, Δ_aff, Δ_xλ, rc, rb, x_m_lvar, uvar_m_x,
                                            s_l, s_u, ilow, iupp, n_cols, n_rows, n_low)
        α_aff_pri = @views compute_α_primal(x, Δ_aff[1:n_cols], FloatData.lvar, FloatData.uvar)
        α_aff_dual_l = @views compute_α_dual(s_l[ilow], Δ_aff[n_rows+n_cols+1: n_rows+n_cols+n_low])
        α_aff_dual_u = @views compute_α_dual(s_u[iupp], Δ_aff[n_rows+n_cols+n_low+1:end])
        # alpha_aff_dual_final is the min of the 2 alpha_aff_dual
        α_aff_dual_final = min(α_aff_dual_l, α_aff_dual_u)
        x_m_l_αΔ_aff .= @views x_m_lvar .+ α_aff_pri .* Δ_aff[1:n_cols][ilow]
        u_m_x_αΔ_aff .= @views uvar_m_x .- α_aff_pri .* Δ_aff[1:n_cols][iupp]
        s_l_αΔ_aff .= @views s_l[ilow] .+ α_aff_dual_final .* Δ_aff[n_rows+n_cols+1: n_rows+n_cols+n_low]
        s_u_αΔ_aff .= @views s_u[iupp] .+ α_aff_dual_final .*  Δ_aff[n_rows+n_cols+n_low+1: end]
        μ_aff = compute_μ(x_m_l_αΔ_aff, u_m_x_αΔ_aff, s_l_αΔ_aff, s_u_αΔ_aff, n_low, n_upp)
        σ = (μ_aff / μ)^3

        # corrector and centering step
        Δ_cc = solve_augmented_system_cc!(J_fact, Δ_cc, Δ_xλ , Δ_aff, σ, μ,x_m_lvar, uvar_m_x,
                                          rxs_l, rxs_u, s_l, s_u, ilow, iupp, n_cols, n_rows, n_low)

        Δ .= Δ_aff .+ Δ_cc # final direction
        α_pri = @views compute_α_primal(x, Δ[1:n_cols], FloatData.lvar, FloatData.uvar)
        α_dual_l = @views compute_α_dual(s_l[ilow], Δ[n_rows+n_cols+1: n_rows+n_cols+n_low])
        α_dual_u = @views compute_α_dual(s_u[iupp], Δ[n_rows+n_cols+n_low+1: end])
        α_dual_final = min(α_dual_l, α_dual_u)

        # new parameters
        x .= @views x .+ α_pri .* Δ[1:n_cols]
        λ .= @views λ .+ α_dual_final .* Δ[n_cols+1: n_rows+n_cols]
        s_l[ilow] .= @views s_l[ilow] .+ α_dual_final .* Δ[n_rows+n_cols+1: n_rows+n_cols+n_low]
        s_u[iupp] .= @views s_u[iupp] .+ α_dual_final .* Δ[n_rows+n_cols+n_low+1: end]
        n_Δx = @views α_pri * norm(Δ[1:n_cols])
        x_m_lvar .= @views x[ilow] .- FloatData.lvar[ilow]
        uvar_m_x .= @views FloatData.uvar[iupp] .- x[iupp]

        if zero(T) in x_m_lvar # "security" if x is too close from lvar ou uvar
            for i=1:n_low
                if x_m_lvar[i] == zero(T)
                    x_m_lvar[i] = eps(T)^2
                end
            end
        end
        if zero(T) in uvar_m_x
            for i=1:n_upp
                if uvar_m_x[i] == zero(T)
                    uvar_m_x[i] = eps(T)^2
                end
            end
        end

        μ = @views compute_μ(x_m_lvar, uvar_m_x, s_l[ilow], s_u[iupp],
                             n_low, n_upp)
        Qx = mul_Qx_COO!(Qx, IntData.Qrows, IntData.Qcols, FloatData.Qvals, x)
        xTQx_2 =  x' * Qx / 2
        ATλ = mul_ATλ_COO!(ATλ, IntData.Arows, IntData.Acols, FloatData.Avals, λ)
        Ax = mul_Ax_COO!(Ax, IntData.Arows, IntData.Acols, FloatData.Avals, x)
        cTx = FloatData.c' * x
        pri_obj = xTQx_2 + cTx + FloatData.c0
        dual_obj = FloatData.b' * λ - xTQx_2 + view(s_l,ilow)'*view(FloatData.lvar, ilow) -
                    view(s_u,iupp)'*view(FloatData.uvar,iupp) + FloatData.c0
        rb .= Ax .- FloatData.b
        rc .= ATλ .-Qx .+ s_l .- s_u .- FloatData.c

        # update stopping criterion values:
        pdd = abs(pri_obj - dual_obj ) / (one(T) + abs(pri_obj))
#         rcNorm, rbNorm = norm(rc), norm(rb)
#         xNorm = norm(x)
#         λNorm = norm(λ)
#         optimal = pdd < ϵ_pdd && rbNorm < ϵ_rb * max(1, bNorm + ANorm * xNorm) &&
#                     rcNorm < ϵ_rc * max(1, cNorm + QNorm * xNorm + ANorm * λNorm)
        rcNorm, rbNorm = norm(rc, Inf), norm(rb, Inf)
        optimal = pdd < ϵ_pdd && rbNorm < tol_rb && rcNorm < tol_rc
        small_Δx, small_μ = n_Δx < tol_Δx, μ < ϵ_μ

        if T == Float32
            k += 1
        else
            k += 4
        end

        l_pdd[k%6+1] = pdd
        mean_pdd = mean(l_pdd)

        if T == Float64 && k > 10  && mean_pdd!=zero(T) && std(l_pdd./mean_pdd) < 1e-2 && c_pdd < 5
            δ_min /= 10
            δ /= 10
            c_pdd += 1
        end
        if T == Float64 && k>10 && c_catch <= 1 &&
                @views minimum(J_augm.nzval[view(diagind_J,1:n_cols)]) < -one(T) / δ / T(1e-6)
            δ /= 10
            δ_min /= 10
            c_pdd += 1
        end
        if T == Float32 && c_pdd < 2 && minimum(J_augm.nzval[view(diagind_J,1:n_cols)]) < -one(T) / δ / T(1e-5)
            break
        end

        if δ >= δ_min
            δ /= 10
        end
        if ρ >= ρ_min
            ρ /= 10
        end

        Δt = time() - start_time
        tired = Δt > max_time

        if display == true
            @info log_row(Any[k, pri_obj, pdd, rbNorm, rcNorm, n_Δx, α_pri, α_dual_final, μ])
        end
    end

    return x, λ, s_l, s_u, x_m_lvar, uvar_m_x, rc, rb,
                rcNorm, rbNorm, Qx, ATλ, Ax, xTQx_2, cTx, pri_obj, dual_obj,
                pdd, l_pdd, mean_pdd, n_Δx, Δt, tired, optimal, μ, k,
                ρ, δ, ρ_min, δ_min, J_augm, J_fact, c_catch, c_pdd
end
