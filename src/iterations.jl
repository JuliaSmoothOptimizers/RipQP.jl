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

function solve_augmented_system_aff!(J_fact, Δ_aff, Δ_xy, rc, rb, x_m_lvar, uvar_m_x,
                                     s_l, s_u, ilow, iupp,  n_cols, n_rows, n_low)

    Δ_xy[1:n_cols] .= .-rc
    Δ_xy[n_cols+1:end] .= .-rb
    Δ_xy[ilow] += @views s_l[ilow]
    Δ_xy[iupp] -= @views s_u[iupp]

    Δ_xy = ldiv!(J_fact, Δ_xy)
    Δ_aff[1:n_cols+n_rows] = Δ_xy
    Δ_aff[n_cols+n_rows+1:n_cols+n_rows+n_low] .= @views .-s_l[ilow] .- s_l[ilow].*Δ_xy[1:n_cols][ilow]./x_m_lvar
    Δ_aff[n_cols+n_rows+n_low+1:end] .= @views .-s_u[iupp] .+ s_u[iupp].*Δ_xy[1:n_cols][iupp]./uvar_m_x
    return Δ_aff
end

function solve_augmented_system_cc!(J_fact, Δ_cc, Δ_xy ,Δ_aff, σ, μ, x_m_lvar, uvar_m_x,
                                    rxs_l, rxs_u, s_l, s_u, ilow, iupp, n_cols, n_rows, n_low)

    rxs_l .= @views (-σ*μ .+ Δ_aff[1:n_cols][ilow].*Δ_aff[n_rows+n_cols+1: n_rows+n_cols+n_low])
    rxs_u .= @views σ*μ .+ Δ_aff[1:n_cols][iupp].*Δ_aff[n_rows+n_cols+n_low+1: end]
    Δ_xy .= zero(eltype(Δ_xy))
    Δ_xy[ilow] .+= rxs_l./x_m_lvar
    Δ_xy[iupp] .+= rxs_u./uvar_m_x

    Δ_xy = ldiv!(J_fact, Δ_xy)
    Δ_cc[1:n_cols+n_rows] = Δ_xy
    Δ_cc[n_cols+n_rows+1:n_cols+n_rows+n_low] .= @views .-(rxs_l.+s_l[ilow].*Δ_xy[1:n_cols][ilow])./x_m_lvar
    Δ_cc[n_cols+n_rows+n_low+1:end] .= @views (rxs_u.+s_u[iupp].*Δ_xy[1:n_cols][iupp])./uvar_m_x
    return Δ_cc
end

function centrality_corr!(Δp, Δm, Δ, Δ_xy, α_p, α_d, J_fact, x, y, s_l, s_u, μ, rxs_l, rxs_u,
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
    Δ_xy .= zero(T)
    Δ_xy[ilow] .+= rxs_l./x_m_lvar
    Δ_xy[iupp] .+= rxs_u./uvar_m_x
    Δ_xy = ldiv!(J_fact, Δ_xy)
    Δm[1:n_cols+n_rows] = Δ_xy
    Δm[n_cols+n_rows+1:n_cols+n_rows+n_low] .= @views .-(rxs_l.+s_l[ilow].*Δ_xy[1:n_cols][ilow])./x_m_lvar
    Δm[n_cols+n_rows+n_low+1:end] .= @views (rxs_u.+s_u[iupp].*Δ_xy[1:n_cols][iupp])./uvar_m_x
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

function iter_mehrotraPC!(pt :: point{T}, itd :: iter_data{T}, fd :: QM_FloatData{T}, id :: QM_IntData,
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
        out = solve_K2!(pt, itd, fd, id, res, regu, pad, cnts)
        if out == one(Int)
            break
        end
        ########################################################################

        pad.Δ .= pad.Δ_aff .+ pad.Δ_cc # final direction
        α_pri = @views compute_α_primal(pt.x, pad.Δ[1:id.n_cols], fd.lvar, fd.uvar)
        α_dual_l = @views compute_α_dual(pt.s_l[id.ilow],
                                         pad.Δ[id.n_rows+id.n_cols+1:id.n_rows+id.n_cols+id.n_low])
        α_dual_u = @views compute_α_dual(pt.s_u[id.iupp], pad.Δ[id.n_rows+id.n_cols+id.n_low+1: end])
        α_dual = min(α_dual_l, α_dual_u)
        
        ############################### centrality corrections ###############################
        if cnts.K > 0
            k_corr = 0
            corr_flag = true #stop correction if false
            pad.Δ_aff .= pad.Δ # for storage issues Δ_aff = Δp  and Δ_cc = Δm
            @inbounds while k_corr < cnts.K && corr_flag
                pad.Δ_aff, pad.Δ, α_pri, α_dual, k_corr,
                    corr_flag = centrality_corr!(pad.Δ_aff, pad.Δ_cc, pad.Δ, pad.Δ_xy, α_pri, α_dual,
                                                 itd.J_fact, pt.x, pt.y, pt.s_l, pt.s_u, itd.μ, pad.rxs_l, pad.rxs_u,
                                                 fd.lvar, fd.uvar, itd.x_m_lvar, itd.uvar_m_x, pad.x_m_l_αΔ_aff,
                                                 pad.u_m_x_αΔ_aff, pad.s_l_αΔ_aff, pad.s_u_αΔ_aff, id.ilow, id.iupp,
                                                 id.n_low, id.n_upp, id.n_rows, id.n_cols, corr_flag, k_corr)
            end
        end
        ######################################################################################

        # new parameters
        pt.x .= @views pt.x .+ α_pri .* pad.Δ[1:id.n_cols]
        pt.y .= @views pt.y .+ α_dual .* pad.Δ[id.n_cols+1: id.n_rows+id.n_cols]
        pt.s_l[id.ilow] .= @views pt.s_l[id.ilow] .+ α_dual .*
                                  pad.Δ[id.n_rows+id.n_cols+1: id.n_rows+id.n_cols+id.n_low]
        pt.s_u[id.iupp] .= @views pt.s_u[id.iupp] .+ α_dual .*
                                  pad.Δ[id.n_rows+id.n_cols+id.n_low+1: end]
        res.n_Δx = @views α_pri * norm(pad.Δ[1:id.n_cols])
        itd.x_m_lvar .= @views pt.x[id.ilow] .- fd.lvar[id.ilow]
        itd.uvar_m_x .= @views fd.uvar[id.iupp] .- pt.x[id.iupp]

        # "security" if x is too close from lvar or uvar
        if zero(T) in itd.x_m_lvar 
            @inbounds @simd for i=1:id.n_low
                if itd.x_m_lvar[i] == zero(T)
                    itd.x_m_lvar[i] = eps(T)^2
                end
            end
        end
        if zero(T) in itd.uvar_m_x
            @inbounds @simd for i=1:id.n_upp
                if itd.uvar_m_x[i] == zero(T)
                    itd.uvar_m_x[i] = eps(T)^2
                end
            end
        end

        # update itd
        itd.μ = @views compute_μ(itd.x_m_lvar, itd.uvar_m_x, pt.s_l[id.ilow], pt.s_u[id.iupp],
                                 id.n_low, id.n_upp)
        itd.Qx = mul!(itd.Qx, Symmetric(fd.Q, :U), pt.x)
        itd.xTQx_2 =  pt.x' * itd.Qx / 2
        itd.ATy = mul!(itd.ATy, fd.AT, pt.y)
        itd.Ax = mul!(itd.Ax, fd.AT', pt.x)
        itd.cTx = fd.c' * pt.x
        itd.pri_obj = itd.xTQx_2 + itd.cTx + fd.c0
        itd.dual_obj = fd.b' * pt.y - itd.xTQx_2 + view(pt.s_l,id.ilow)'*view(fd.lvar, id.ilow) -
                        view(pt.s_u, id.iupp)'*view(fd.uvar, id.iupp) + fd.c0
        res.rb .= itd.Ax .- fd.b
        res.rc .= itd.ATy .- itd.Qx .+ pt.s_l .- pt.s_u .- fd.c

        # update stopping criterion values:
        itd.pdd = abs(itd.pri_obj - itd.dual_obj ) / (one(T) + abs(itd.pri_obj))
#         rcNorm, rbNorm = norm(rc), norm(rb)
#         xNorm = norm(x)
#         yNorm = norm(y)
#         optimal = pdd < ϵ_pdd && rbNorm < ϵ_rb * max(1, bNorm + ANorm * xNorm) &&
#                     rcNorm < ϵ_rc * max(1, cNorm + QNorm * xNorm + ANorm * yNorm)
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

        # update ρ and δ values, check J_augm diag magnitude
        if regu.regul == :classic  
            itd.l_pdd[cnts.k%6+1] = itd.pdd
            itd.mean_pdd = mean(itd.l_pdd)

            if T == Float64 && cnts.k > 10  && itd.mean_pdd!=zero(T) && std(itd.l_pdd./itd.mean_pdd) < 1e-2 && cnts.c_pdd < 5
                regu.δ_min /= 10
                regu.δ /= 10
                cnts.c_pdd += 1
            end
            if T == Float64 && cnts.k>10 && cnts.c_catch <= 1 &&
                    @views minimum(itd.J_augm.nzval[view(itd.diagind_J,1:id.n_cols)]) < -one(T) / regu.δ / T(1e-6)
                regu.δ /= 10
                regu.δ_min /= 10
                cnts.c_pdd += 1
            elseif T != T0 && cnts.c_pdd < 2 &&
                    @views minimum(itd.J_augm.nzval[view(itd.diagind_J,1:id.n_cols)]) < -one(T) / regu.δ / T(1e-5)
                break
            elseif T == Float128 && cnts.k>10 && cnts.c_catch <= 1 &&
                    @views minimum(itd.J_augm.nzval[view(itd.diagind_J,1:id.n_cols)]) < -one(T) / regu.δ / T(1e-15)
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
end
