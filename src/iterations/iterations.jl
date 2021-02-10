include("centrality_corr.jl")
include("regularization.jl")
include("direct_methods/K2.jl")

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

@inline function compute_αs(x, s_l, s_u, lvar, uvar, Δ, n_low, n_rows, n_cols)
    α_pri = @views compute_α_primal(x, Δ[1:n_cols], lvar, uvar)
    α_dual_l = @views compute_α_dual(s_l, Δ[n_rows+n_cols+1:n_rows+n_cols+n_low])
    α_dual_u = @views compute_α_dual(s_u, Δ[n_rows+n_cols+n_low+1: end])
    return α_pri, min(α_dual_l, α_dual_u)
end

@inline function compute_μ(x_m_lvar, uvar_m_x, s_l, s_u, nb_low, nb_upp)
    return (dot(s_l, x_m_lvar) + dot(s_u, uvar_m_x)) / (nb_low + nb_upp)
end

function solve_augmented_system_aff!(Δ_aff, J_fact, Δ_xy, rc, rb, x_m_lvar, uvar_m_x,
                                     s_l, s_u, ilow, iupp,  n_cols, n_rows, n_low)

    Δ_xy[1:n_cols] .= .-rc
    Δ_xy[n_cols+1:end] .= .-rb
    Δ_xy[ilow] .+= s_l
    Δ_xy[iupp] .-= s_u

    Δ_xy = ldiv!(J_fact, Δ_xy)
    Δ_aff[1:n_cols+n_rows] = Δ_xy
    Δ_aff[n_cols+n_rows+1:n_cols+n_rows+n_low] .= @views .-s_l .- s_l.*Δ_xy[ilow]./x_m_lvar
    Δ_aff[n_cols+n_rows+n_low+1:end] .= @views .-s_u .+ s_u.*Δ_xy[iupp]./uvar_m_x
end

function update_pt_aff!(x_m_l_αΔ_aff, u_m_x_αΔ_aff, s_l_αΔ_aff, s_u_αΔ_aff, Δ_aff, x_m_lvar, uvar_m_x, 
                        s_l, s_u, α_aff_pri, α_aff_dual, ilow, iupp, n_low, n_rows, n_cols)
    x_m_l_αΔ_aff .= @views x_m_lvar .+ α_aff_pri .* Δ_aff[ilow]
    u_m_x_αΔ_aff .= @views uvar_m_x .- α_aff_pri .* Δ_aff[iupp]
    s_l_αΔ_aff .= @views s_l .+ α_aff_dual .* Δ_aff[n_rows+n_cols+1: n_rows+n_cols+n_low]
    s_u_αΔ_aff .= @views s_u .+ α_aff_dual .* Δ_aff[n_rows+n_cols+n_low+1: end]
end

function solve_augmented_system_cc!(Δ_cc, J_fact, Δ_xy, x_m_lvar, uvar_m_x, rxs_l, rxs_u, 
                                    s_l, s_u, ilow, iupp, n_cols, n_rows, n_low)

    Δ_xy .= 0
    Δ_xy[ilow] .+= rxs_l./x_m_lvar
    Δ_xy[iupp] .+= rxs_u./uvar_m_x

    Δ_xy = ldiv!(J_fact, Δ_xy)
    Δ_cc[1:n_cols+n_rows] = Δ_xy
    Δ_cc[n_cols+n_rows+1:n_cols+n_rows+n_low] .= @views .-(rxs_l.+s_l.*Δ_xy[ilow])./x_m_lvar
    Δ_cc[n_cols+n_rows+n_low+1:end] .= @views (rxs_u.+s_u.*Δ_xy[iupp])./uvar_m_x
end

function update_pt!(x, y, s_l, s_u, α_pri, α_dual, Δ, n_low, n_rows, n_cols)
    x .= @views x .+ α_pri .* Δ[1:n_cols]
    y .= @views y .+ α_dual .* Δ[n_cols+1: n_rows+n_cols]
    s_l .= @views s_l .+ α_dual .* Δ[n_rows+n_cols+1: n_rows+n_cols+n_low]
    s_u .= @views s_u .+ α_dual .* Δ[n_rows+n_cols+n_low+1: end]
end

# "security" if x is too close from lvar or uvar
function boundary_safety!(x_m_lvar, uvar_m_x, n_low, n_upp, T) 
    if 0 in x_m_lvar 
        @inbounds @simd for i=1:n_low
            if x_m_lvar[i] == 0
                x_m_lvar[i] = eps(T)^2
            end
        end
    end
    if 0 in uvar_m_x
        @inbounds @simd for i=1:n_upp
            if uvar_m_x[i] == 0
                uvar_m_x[i] = eps(T)^2
            end
        end
    end
end

function update_iter_data!(itd, pt, fd, id, safety) 

    T = eltype(itd.x_m_lvar)
    itd.x_m_lvar .= @views pt.x[id.ilow] .- fd.lvar[id.ilow]
    itd.uvar_m_x .= @views fd.uvar[id.iupp] .- pt.x[id.iupp]
    safety && boundary_safety!(itd.x_m_lvar, itd.uvar_m_x, id.n_low, id.n_upp, T)

    itd.μ = compute_μ(itd.x_m_lvar, itd.uvar_m_x, pt.s_l, pt.s_u, id.n_low, id.n_upp)
    itd.Qx = mul!(itd.Qx, Symmetric(fd.Q, :U), pt.x)
    itd.xTQx_2 =  dot(pt.x, itd.Qx) / 2
    itd.ATy = mul!(itd.ATy, fd.AT, pt.y)
    itd.Ax = mul!(itd.Ax, fd.AT', pt.x)
    itd.cTx = dot(fd.c, pt.x)
    itd.pri_obj = itd.xTQx_2 + itd.cTx + fd.c0
    itd.dual_obj = @views dot(fd.b, pt.y) - itd.xTQx_2 + dot(pt.s_l, fd.lvar[id.ilow]) -
                        dot(pt.s_u, fd.uvar[id.iupp]) + fd.c0  
    itd.pdd = abs(itd.pri_obj - itd.dual_obj ) / (one(T) + abs(itd.pri_obj))                     
end

function update_residuals!(res, s_l, s_u, ilow, iupp, Δ, Ax, ATy, Qx, b, c, α_pri, n_cols) 
    res.n_Δx = @views α_pri * norm(Δ[1:n_cols])
    res.rb .= Ax .- b
    res.rc .= ATy .- Qx .- c
    res.rc[ilow] .+= s_l
    res.rc[iupp] .-= s_u
    # update stopping criterion values:
#         rcNorm, rbNorm = norm(rc), norm(rb)
#         xNorm = norm(x)
#         yNorm = norm(y)
#         optimal = pdd < ϵ_pdd && rbNorm < ϵ_rb * max(1, bNorm + ANorm * xNorm) &&
#                     rcNorm < ϵ_rc * max(1, cNorm + QNorm * xNorm + ANorm * yNorm)
    res.rcNorm, res.rbNorm = norm(res.rc, Inf), norm(res.rb, Inf)
end

function iter_mehrotraPC!(pt :: point{T}, itd :: iter_data{T}, fd :: QM_FloatData{T}, id :: QM_IntData,
                          res :: residuals{T}, sc :: stop_crit{Tc}, pad :: preallocated_data{T}, 
                          ϵ :: tolerances{T}, cnts :: counters, T0 :: DataType, 
                          display :: Bool) where {T<:Real, Tc<:Real}
    
    if itd.regu.regul == :dynamic
        itd.J_fact.r1, itd.J_fact.r2 = -T(eps(T)^(3/4)), T(eps(T)^(1/2))
    elseif itd.regu.regul == :none
        itd.regu.ρ, itd.regu.δ = zero(T), zero(T)
    end
    @inbounds while cnts.k < sc.max_iter && !sc.optimal && !sc.tired # && !small_μ && !small_μ

        # Solve system to find a direction of descent 
        out = solve_K2!(pt, itd, fd, id, res, pad, cnts, T, T0)
        out == 1 && break

        α_pri, α_dual = compute_αs(pt.x, pt.s_l, pt.s_u, fd.lvar, fd.uvar, pad.Δ, id.n_low, 
                                   id.n_rows, id.n_cols)
        
        if cnts.K > 0   # centrality corrections
            α_pri, α_dual = multi_centrality_corr!(pad, pt, α_pri, α_dual, itd.J_fact, itd.μ, 
                                                   fd.lvar, fd.uvar, itd.x_m_lvar, itd.uvar_m_x, 
                                                   id, cnts.K, T)
            ## TODO replace by centrality_corr.jl, deal with α
        end

        # (x, y, s_l, s_u) += α * Δ
        update_pt!(pt.x, pt.y, pt.s_l, pt.s_u, α_pri, α_dual, pad.Δ, id.n_low, id.n_rows, id.n_cols)
        # update data and residuals after the new point is computed
        update_iter_data!(itd, pt, fd, id, true)
        update_residuals!(res, pt.s_l, pt.s_u, id.ilow, id.iupp, pad.Δ, itd.Ax, itd.ATy, itd.Qx, 
                          fd.b, fd.c, α_pri, id.n_cols)

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

        if itd.regu.regul == :classic  # update ρ and δ values, check J_augm diag magnitude 
            out = update_regu_diagJ!(itd.regu, itd.J_augm.nzval, itd.diagind_J, id.n_cols, itd.pdd, 
                                     itd.l_pdd, itd.mean_pdd, cnts, T, T0) 
            out == 1 && break
        end

        sc.Δt = time() - sc.start_time
        sc.tired = sc.Δt > sc.max_time

        if display == true
            @info log_row(Any[cnts.k, itd.pri_obj, itd.pdd, res.rbNorm, res.rcNorm, res.n_Δx, α_pri, α_dual, itd.μ, itd.regu.ρ, itd.regu.δ])
        end
    end
end
