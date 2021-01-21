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
    Δ_aff[n_cols+n_rows+1:n_cols+n_rows+n_low] .= @views .-s_l[ilow] .- s_l[ilow].*Δ_xλ[1:n_cols][ilow]./x_m_lvar
    Δ_aff[n_cols+n_rows+n_low+1:end] .= @views .-s_u[iupp] .+ s_u[iupp].*Δ_xλ[1:n_cols][iupp]./uvar_m_x
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

function centrality_corr!(Δp, Δm, Δ, Δ_xλ, α_p, α_d, J_fact, x, λ, s_l, s_u, μ, rxs_l, rxs_u,
                          lvar, uvar, x_m_lvar, uvar_m_x, x_m_l_αΔp, u_m_x_αΔp, s_l_αΔp, s_u_αΔp,
                          ilow, iupp, n_low, n_upp, n_rows, n_cols, corr_flag, k_corr)
    T = eltype(Δ) # Δp = Δ_aff + Δ_cc
    δα, γ, βmin, βmax = T(0.1), T(0.1), T(0.1), T(10)
    α_p2, α_d2 = min(α_p + δα, one(T)), min(α_d + δα, one(T))
    x_m_l_αΔp .= @views x_m_lvar .+ α_p2 .* Δp[1:n_cols][ilow]
    u_m_x_αΔp .= @views uvar_m_x .- α_p2 .* Δp[1:n_cols][iupp]
    s_l_αΔp .= @views s_l[ilow] .+ α_d2 .* Δp[n_rows+n_cols+1: n_rows+n_cols+n_low]
    s_u_αΔp .= @views s_u[iupp] .+ α_d2 .* Δp[n_rows+n_cols+n_low+1: end]
    μ_p = compute_μ(x_m_l_αΔp, u_m_x_αΔp, s_l_αΔp, s_u_αΔp, n_low, n_upp)
    σ = (μ_p / μ)^3
    Hmin, Hmax = βmin * σ * μ, βmax * σ * μ

    # compute Δm
    @inbounds @simd for i=1:n_low
        rxs_l[i] = s_l_αΔp[i] * x_m_l_αΔp[i]
        if Hmin <= rxs_l[i] <= Hmax
            rxs_l[i] = zero(T)
        elseif rxs_l[i] < Hmin
            rxs_l[i] -= Hmin
        else
            rxs_l[i] -= Hmax
        end
        if rxs_l[i] > Hmax
            rxs_l[i] = Hmax
        end
    end
    @inbounds @simd for i =1:n_upp
        rxs_u[i] = -s_u_αΔp[i]*u_m_x_αΔp[i]
        if Hmin <= -rxs_u[i] <= Hmax
            rxs_u[i] = zero(T)
        elseif -rxs_u[i] < Hmin
            rxs_u[i] += Hmin
        else
            rxs_u[i] += Hmax
        end
        if rxs_u[i] < -Hmax
            rxs_u[i] = -Hmax
        end
    end
    Δ_xλ .= zero(T)
    Δ_xλ[ilow] .+= rxs_l./x_m_lvar
    Δ_xλ[iupp] .+= rxs_u./uvar_m_x
    Δ_xλ = ldiv!(J_fact, Δ_xλ)
    Δm[1:n_cols+n_rows] = Δ_xλ
    Δm[n_cols+n_rows+1:n_cols+n_rows+n_low] .= @views .-(rxs_l.+s_l[ilow].*Δ_xλ[1:n_cols][ilow])./x_m_lvar
    Δm[n_cols+n_rows+n_low+1:end] .= @views (rxs_u.+s_u[iupp].*Δ_xλ[1:n_cols][iupp])./uvar_m_x
    Δ .= Δp .+ Δm
    α_p2 = @views compute_α_primal(x, Δ[1:n_cols], lvar, uvar)
    α_d_l2 = @views compute_α_dual(s_l[ilow], Δ[n_rows+n_cols+1:n_rows+n_cols+n_low])
    α_d_u2 = @views compute_α_dual(s_u[iupp], Δ[n_rows+n_cols+n_low+1:end])
    α_d2 = min(α_d_l2, α_d_u2)

    if α_p2 >= α_p + γ*δα && α_d2 >= α_d + γ*δα
        k_corr += 1
        Δp .= Δ
        α_p, α_d = α_p2, α_d2
    else
        Δ .= Δp
        corr_flag = false
    end

    return Δp, Δ, α_p, α_d, k_corr, corr_flag
end

function iter_mehrotraPC!(pt :: point{T}, itd :: iter_data{T}, FloatData :: QM_FloatData{T}, IntData :: QM_IntData,
                          res :: residuals{T}, sc :: stop_crit, Δt :: Real, regu :: regularization{T},
                          pad :: preallocated_data{T}, max_iter :: Int, ϵ :: tolerances{T}, start_time :: Real,
                          max_time :: Real, cnts :: counters, T0 :: DataType, display :: Bool) where {T<:Real}
    if regu.regul == :dynamic
        regu.ρ, regu.δ = T(eps(T)^(3/4)), T(eps(T)^(1/2))
    elseif regu.regul == :none
        regu.ρ, regu.δ = zero(T), zero(T)
    end
    @inbounds while cnts.k<max_iter && !sc.optimal && !sc.tired # && !small_μ && !small_μ

        ###################### J update and factorization ######################
        if regu.regul == :classic
            itd.tmp_diag .= -regu.ρ
            itd.J_augm.nzval[view(itd.diagind_J, IntData.n_cols+1:IntData.n_rows+IntData.n_cols)] .= regu.δ
        else
            itd.tmp_diag .= zero(T)
        end
        itd.tmp_diag[IntData.ilow] .-= @views pt.s_l[IntData.ilow] ./ itd.x_m_lvar
        itd.tmp_diag[IntData.iupp] .-= @views pt.s_u[IntData.iupp] ./ itd.uvar_m_x
        itd.tmp_diag[itd.diag_Q.nzind] .-= itd.diag_Q.nzval
        itd.J_augm.nzval[view(itd.diagind_J,1:IntData.n_cols)] = itd.tmp_diag 
        if regu.regul == :dynamic
            Amax = @views norm(itd.J_augm.nzval[itd.diagind_J], Inf)
            if Amax > T(1e6) / regu.δ && cnts.c_pdd < 8
                if T == Float32
                    break
                elseif length(FloatData.Q.nzval) > 0 || cnts.c_pdd < 4
                    cnts.c_pdd += 1
                    regu.δ /= 10
                    # regu.ρ /= 10
                end
            end
            itd.J_fact = ldl_factorize!(Symmetric(itd.J_augm, :U), itd.J_fact,
                                        tol=Amax*T(eps(T)), r1=-regu.ρ, r2=regu.δ, n_d=IntData.n_cols)
        elseif regu.regul == :classic
            itd.J_fact = try ldl_factorize!(Symmetric(itd.J_augm, :U), itd.J_fact)
            catch
                if T == Float32
                    break
                    break
                elseif T0 == Float128 && T == Float64
                    break
                    break
                end
                if cnts.c_pdd == 0 && cnts.c_catch == 0
                    regu.δ *= T(1e2)
                    regu.δ_min *= T(1e2)
                    regu.ρ *= T(1e5)
                    regu.ρ_min *= T(1e5)
                elseif cnts.c_pdd == 0 && cnts.c_catch != 0
                    regu.δ *= T(1e1)
                    regu.δ_min *= T(1e1)
                    regu.ρ *= T(1e0)
                    regu.ρ_min *= T(1e0)
                elseif cnts.c_pdd != 0 && cnts.c_catch==0
                    regu.δ *= T(1e5)
                    regu.δ_min *= T(1e5)
                    regu.ρ *= T(1e5)
                    regu.ρ_min *= T(1e5)
                else
                    regu.δ *= T(1e1)
                    regu.δ_min *= T(1e1)
                    regu.ρ *= T(1e1)
                    regu.ρ_min *= T(1e1)
                end
                cnts.c_catch += 1
                itd.tmp_diag .= -regu.ρ
                itd.tmp_diag[IntData.ilow] .-= @views pt.s_l[IntData.ilow] ./ itd.x_m_lvar
                itd.tmp_diag[IntData.iupp] .-= @views pt.s_u[IntData.iupp] ./ itd.uvar_m_x
                itd.tmp_diag[itd.diag_Q.nzind] .-= itd.diag_Q.nzval
                itd.J_augm.nzval[view(itd.diagind_J,1:IntData.n_cols)] = itd.tmp_diag 
                itd.J_augm.nzval[view(itd.diagind_J, IntData.n_cols+1:IntData.n_rows+IntData.n_cols)] .= regu.δ
                itd.J_fact = ldl_factorize!(Symmetric(itd.J_augm, :U), itd.J_fact)
            end
        else # no regularization
            itd.J_fact = ldl_factorize!(Symmetric(itd.J_augm, :U), itd.J_fact)
        end

        if cnts.c_catch >= 4
            break
        end
        ########################################################################

        pad.Δ_aff = solve_augmented_system_aff!(itd.J_fact, pad.Δ_aff, pad.Δ_xλ, res.rc, res.rb,
                                                itd.x_m_lvar, itd.uvar_m_x, pt.s_l, pt.s_u,
                                                IntData.ilow, IntData.iupp, IntData.n_cols, IntData.n_rows,
                                                IntData.n_low)
        α_aff_pri = @views compute_α_primal(pt.x, pad.Δ_aff[1:IntData.n_cols], FloatData.lvar, FloatData.uvar)
        α_aff_dual_l = @views compute_α_dual(pt.s_l[IntData.ilow],
                                             pad.Δ_aff[IntData.n_rows+IntData.n_cols+1:IntData.n_rows+IntData.n_cols+IntData.n_low])
        α_aff_dual_u = @views compute_α_dual(pt.s_u[IntData.iupp],
                                             pad.Δ_aff[IntData.n_rows+IntData.n_cols+IntData.n_low+1:end])
        # alpha_aff_dual is the min of the 2 alpha_aff_dual
        α_aff_dual = min(α_aff_dual_l, α_aff_dual_u)
        pad.x_m_l_αΔ_aff .= @views itd.x_m_lvar .+ α_aff_pri .* pad.Δ_aff[1:IntData.n_cols][IntData.ilow]
        pad.u_m_x_αΔ_aff .= @views itd.uvar_m_x .- α_aff_pri .* pad.Δ_aff[1:IntData.n_cols][IntData.iupp]
        pad.s_l_αΔ_aff .= @views pt.s_l[IntData.ilow] .+ α_aff_dual .*
                            pad.Δ_aff[IntData.n_rows+IntData.n_cols+1: IntData.n_rows+IntData.n_cols+IntData.n_low]
        pad.s_u_αΔ_aff .= @views pt.s_u[IntData.iupp] .+ α_aff_dual .*
                            pad.Δ_aff[IntData.n_rows+IntData.n_cols+IntData.n_low+1: end]
        μ_aff = compute_μ(pad.x_m_l_αΔ_aff, pad.u_m_x_αΔ_aff, pad.s_l_αΔ_aff, pad.s_u_αΔ_aff,
                          IntData.n_low, IntData.n_upp)
        σ = (μ_aff / itd.μ)^3

        # corrector and centering step
        pad.Δ_cc = solve_augmented_system_cc!(itd.J_fact, pad.Δ_cc, pad.Δ_xλ , pad.Δ_aff, σ, itd.μ,
                                              itd.x_m_lvar, itd.uvar_m_x, pad.rxs_l, pad.rxs_u, pt.s_l, pt.s_u,
                                              IntData.ilow, IntData.iupp, IntData.n_cols, IntData.n_rows,
                                              IntData.n_low)
        pad.Δ .= pad.Δ_aff .+ pad.Δ_cc # final direction
        α_pri = @views compute_α_primal(pt.x, pad.Δ[1:IntData.n_cols], FloatData.lvar, FloatData.uvar)
        α_dual_l = @views compute_α_dual(pt.s_l[IntData.ilow],
                                         pad.Δ[IntData.n_rows+IntData.n_cols+1:IntData.n_rows+IntData.n_cols+IntData.n_low])
        α_dual_u = @views compute_α_dual(pt.s_u[IntData.iupp], pad.Δ[IntData.n_rows+IntData.n_cols+IntData.n_low+1: end])
        α_dual = min(α_dual_l, α_dual_u)
        ############################### centrality corrections ###############################
        if cnts.K > 0
            k_corr = 0
            corr_flag = true #stop correction if false
            pad.Δ_aff .= pad.Δ # for storage issues Δ_aff = Δp  and Δ_cc = Δm
            @inbounds while k_corr < cnts.K && corr_flag
                pad.Δ_aff, pad.Δ, α_pri, α_dual, k_corr,
                    corr_flag = centrality_corr!(pad.Δ_aff, pad.Δ_cc, pad.Δ, pad.Δ_xλ, α_pri, α_dual,
                                                 itd.J_fact, pt.x, pt.λ, pt.s_l, pt.s_u, itd.μ, pad.rxs_l, pad.rxs_u,
                                                 FloatData.lvar, FloatData.uvar, itd.x_m_lvar, itd.uvar_m_x, pad.x_m_l_αΔ_aff,
                                                 pad.u_m_x_αΔ_aff, pad.s_l_αΔ_aff, pad.s_u_αΔ_aff, IntData.ilow, IntData.iupp,
                                                 IntData.n_low, IntData.n_upp, IntData.n_rows, IntData.n_cols, corr_flag, k_corr)
            end
        end
        ######################################################################################
        # new parameters
        pt.x .= @views pt.x .+ α_pri .* pad.Δ[1:IntData.n_cols]
        pt.λ .= @views pt.λ .+ α_dual .* pad.Δ[IntData.n_cols+1: IntData.n_rows+IntData.n_cols]
        pt.s_l[IntData.ilow] .= @views pt.s_l[IntData.ilow] .+ α_dual .*
                                  pad.Δ[IntData.n_rows+IntData.n_cols+1: IntData.n_rows+IntData.n_cols+IntData.n_low]
        pt.s_u[IntData.iupp] .= @views pt.s_u[IntData.iupp] .+ α_dual .*
                                  pad.Δ[IntData.n_rows+IntData.n_cols+IntData.n_low+1: end]
        res.n_Δx = @views α_pri * norm(pad.Δ[1:IntData.n_cols])
        itd.x_m_lvar .= @views pt.x[IntData.ilow] .- FloatData.lvar[IntData.ilow]
        itd.uvar_m_x .= @views FloatData.uvar[IntData.iupp] .- pt.x[IntData.iupp]

        if zero(T) in itd.x_m_lvar # "security" if x is too close from lvar ou uvar
            @inbounds @simd for i=1:IntData.n_low
                if itd.x_m_lvar[i] == zero(T)
                    itd.x_m_lvar[i] = eps(T)^2
                end
            end
        end
        if zero(T) in itd.uvar_m_x
            @inbounds @simd for i=1:IntData.n_upp
                if itd.uvar_m_x[i] == zero(T)
                    itd.uvar_m_x[i] = eps(T)^2
                end
            end
        end

        itd.μ = @views compute_μ(itd.x_m_lvar, itd.uvar_m_x, pt.s_l[IntData.ilow], pt.s_u[IntData.iupp],
                                 IntData.n_low, IntData.n_upp)
        itd.Qx = mul!(itd.Qx, Symmetric(FloatData.Q, :U), pt.x)
        itd.xTQx_2 =  pt.x' * itd.Qx / 2
        itd.ATλ = mul!(itd.ATλ, FloatData.A, pt.λ)
        itd.Ax = mul!(itd.Ax, FloatData.A', pt.x)
        itd.cTx = FloatData.c' * pt.x
        itd.pri_obj = itd.xTQx_2 + itd.cTx + FloatData.c0
        itd.dual_obj = FloatData.b' * pt.λ - itd.xTQx_2 + view(pt.s_l,IntData.ilow)'*view(FloatData.lvar, IntData.ilow) -
                        view(pt.s_u, IntData.iupp)'*view(FloatData.uvar, IntData.iupp) + FloatData.c0
        res.rb .= itd.Ax .- FloatData.b
        res.rc .= itd.ATλ .- itd.Qx .+ pt.s_l .- pt.s_u .- FloatData.c

        # update stopping criterion values:
        itd.pdd = abs(itd.pri_obj - itd.dual_obj ) / (one(T) + abs(itd.pri_obj))
#         rcNorm, rbNorm = norm(rc), norm(rb)
#         xNorm = norm(x)
#         λNorm = norm(λ)
#         optimal = pdd < ϵ_pdd && rbNorm < ϵ_rb * max(1, bNorm + ANorm * xNorm) &&
#                     rcNorm < ϵ_rc * max(1, cNorm + QNorm * xNorm + ANorm * λNorm)
        res.rcNorm, res.rbNorm = norm(res.rc, Inf), norm(res.rb, Inf)
        sc.optimal = itd.pdd < ϵ.pdd && res.rbNorm < ϵ.tol_rb && res.rcNorm < ϵ.tol_rc
        sc.small_Δx, sc.small_μ = res.n_Δx < ϵ.Δx, itd.μ < ϵ.μ

        cnts.k += 1
        if T == Float32
            cnts.km += 1
        elseif T == Float64
            cnts.km += 4
        else
            cnts.km += 16
        end

        if regu.regul == :classic  # update ρ and δ values, check J_augm diag magnitude
            itd.l_pdd[cnts.k%6+1] = itd.pdd
            itd.mean_pdd = mean(itd.l_pdd)

            if T == Float64 && cnts.k > 10  && itd.mean_pdd!=zero(T) && std(itd.l_pdd./itd.mean_pdd) < 1e-2 && cnts.c_pdd < 5
                regu.δ_min /= 10
                regu.δ /= 10
                cnts.c_pdd += 1
            end
            if T == Float64 && cnts.k>10 && cnts.c_catch <= 1 &&
                    @views minimum(itd.J_augm.nzval[view(itd.diagind_J,1:IntData.n_cols)]) < -one(T) / regu.δ / T(1e-6)
                regu.δ /= 10
                regu.δ_min /= 10
                cnts.c_pdd += 1
            elseif T != T0 && cnts.c_pdd < 2 &&
                    @views minimum(itd.J_augm.nzval[view(itd.diagind_J,1:IntData.n_cols)]) < -one(T) / regu.δ / T(1e-5)
                break
            elseif T == Float128 && cnts.k>10 && cnts.c_catch <= 1 &&
                    @views minimum(itd.J_augm.nzval[view(itd.diagind_J,1:IntData.n_cols)]) < -one(T) / regu.δ / T(1e-15)
                regu.δ /= 10
                regu.δ_min /= 10
                cnts.c_pdd += 1
            end

            if regu.δ >= regu.δ_min
                regu.δ /= 10
            end
            if regu.ρ >= regu.ρ_min
                regu.ρ /= 10
            end
        end

        Δt = time() - start_time
        sc.tired = Δt > max_time

        if display == true
            @info log_row(Any[cnts.k, itd.pri_obj, itd.pdd, res.rbNorm, res.rcNorm, res.n_Δx, α_pri, α_dual, itd.μ, regu.ρ, regu.δ])
        end
    end

    return pt, res, itd, Δt, sc, cnts, regu
end
