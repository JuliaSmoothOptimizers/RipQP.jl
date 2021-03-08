include("centrality_corr.jl")
include("regularization.jl")
include("direct_methods/K2.jl")
include("direct_methods/K2_5.jl")

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

@inline function compute_αs(x, s_l, s_u, lvar, uvar, Δxy, Δs_l, Δs_u, n_cols)
    α_pri = @views compute_α_primal(x, Δxy[1:n_cols], lvar, uvar)
    α_dual_l = compute_α_dual(s_l, Δs_l)
    α_dual_u = compute_α_dual(s_u, Δs_u)
    return α_pri, min(α_dual_l, α_dual_u)
end

@inline function compute_μ(x_m_lvar, uvar_m_x, s_l, s_u, nb_low, nb_upp)
    return (dot(s_l, x_m_lvar) + dot(s_u, uvar_m_x)) / (nb_low + nb_upp)
end

function solve_augmented_system_aff!(Δxy_aff, Δs_l_aff, Δs_u_aff, J_fact, rc, rb, x_m_lvar, uvar_m_x,
                                     s_l, s_u, ilow, iupp, n_cols)

    Δxy_aff[1:n_cols] .= .-rc
    Δxy_aff[n_cols+1:end] .= .-rb
    Δxy_aff[ilow] .+= s_l
    Δxy_aff[iupp] .-= s_u

    ldiv!(J_fact, Δxy_aff)
    Δs_l_aff .= @views .-s_l .- s_l.*Δxy_aff[ilow]./x_m_lvar
    Δs_u_aff .= @views .-s_u .+ s_u.*Δxy_aff[iupp]./uvar_m_x
end

function update_pt_aff!(x_m_l_αΔ_aff, u_m_x_αΔ_aff, s_l_αΔ_aff, s_u_αΔ_aff, Δxy_aff, Δs_l_aff, Δs_u_aff, 
                        x_m_lvar, uvar_m_x, s_l, s_u, α_aff_pri, α_aff_dual, ilow, iupp)
    x_m_l_αΔ_aff .= @views x_m_lvar .+ α_aff_pri .* Δxy_aff[ilow]
    u_m_x_αΔ_aff .= @views uvar_m_x .- α_aff_pri .* Δxy_aff[iupp]
    s_l_αΔ_aff .= s_l .+ α_aff_dual .* Δs_l_aff
    s_u_αΔ_aff .= s_u .+ α_aff_dual .* Δs_u_aff
end

function solve_augmented_system_cc!(J_fact, Δxy_cc, Δs_l_cc, Δs_u_cc, x_m_lvar, uvar_m_x, rxs_l, rxs_u, 
                                    s_l, s_u, ilow, iupp)

    Δxy_cc .= 0
    Δxy_cc[ilow] .+= rxs_l./x_m_lvar
    Δxy_cc[iupp] .+= rxs_u./uvar_m_x

    ldiv!(J_fact, Δxy_cc)
    Δs_l_cc .= @views .-(rxs_l.+s_l.*Δxy_cc[ilow])./x_m_lvar
    Δs_u_cc .= @views (rxs_u.+s_u.*Δxy_cc[iupp])./uvar_m_x
end

function update_pt!(x, y, s_l, s_u, α_pri, α_dual, Δxy, Δs_l, Δs_u, n_rows, n_cols)
    x .= @views x .+ α_pri .* Δxy[1:n_cols]
    y .= @views y .+ α_dual .* Δxy[n_cols+1: n_rows+n_cols]
    s_l .= s_l .+ α_dual .* Δs_l
    s_u .= s_u .+ α_dual .* Δs_u
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

function update_data!(pt :: point{T}, α_pri :: T, α_dual :: T, itd :: iter_data{T}, pad :: preallocated_data{T}, 
                      res :: residuals{T}, fd :: QM_FloatData{T}, id :: QM_IntData) where {T<:Real}

    # (x, y, s_l, s_u) += α * Δ
    update_pt!(pt.x, pt.y, pt.s_l, pt.s_u, α_pri, α_dual, pad.Δxy, pad.Δs_l, pad.Δs_u, id.n_rows, id.n_cols)
    update_iter_data!(itd, pt, fd, id, true)
    
    #update residuals
    res.n_Δx = @views α_pri * norm(pad.Δxy[1:id.n_cols])
    res.rb .= itd.Ax .- fd.b
    res.rc .= itd.ATy .- itd.Qx .- fd.c
    res.rc[id.ilow] .+= pt.s_l
    res.rc[id.iupp] .-= pt.s_u
    # update stopping criterion values:
#         rcNorm, rbNorm = norm(rc), norm(rb)
#         xNorm = norm(x)
#         yNorm = norm(y)
#         optimal = pdd < ϵ_pdd && rbNorm < ϵ_rb * max(1, bNorm + ANorm * xNorm) &&
#                     rcNorm < ϵ_rc * max(1, cNorm + QNorm * xNorm + ANorm * yNorm)
    res.rcNorm, res.rbNorm = norm(res.rc, Inf), norm(res.rb, Inf)
end

function iter!(pt :: point{T}, itd :: iter_data{T}, fd :: Abstract_QM_FloatData{T}, id :: QM_IntData, res :: residuals{T}, 
               sc :: stop_crit{Tc}, pad :: preallocated_data{T}, ϵ :: tolerances{T}, solve! :: Function, 
               cnts :: counters, T0 :: DataType, display :: Bool) where {T<:Real, Tc<:Real}
    
    if itd.regu.regul == :dynamic
        itd.regu.ρ, itd.regu.δ = -T(eps(T)^(3/4)), T(eps(T)^(0.45))
        itd.J_fact.r1, itd.J_fact.r2 = itd.regu.ρ, itd.regu.δ
    elseif itd.regu.regul == :none
        itd.regu.ρ, itd.regu.δ = zero(T), zero(T)
    end
    @inbounds while cnts.k < sc.max_iter && !sc.optimal && !sc.tired # && !small_μ && !small_μ

        # Solve system to find a direction of descent 
        out = solve!(pt, itd, fd, id, res, pad, cnts, T, T0)
        out == 1 && break

        α_pri, α_dual = compute_αs(pt.x, pt.s_l, pt.s_u, fd.lvar, fd.uvar, pad.Δxy, pad.Δs_l, pad.Δs_u, id.n_cols)
        
        if cnts.K > 0   # centrality corrections
            α_pri, α_dual = multi_centrality_corr!(pad, pt, α_pri, α_dual, itd.J_fact, itd.μ, 
                                                   fd.lvar, fd.uvar, itd.x_m_lvar, itd.uvar_m_x, 
                                                   id, cnts.K, T)
            ## TODO replace by centrality_corr.jl, deal with α
        end

        update_data!(pt, α_pri, α_dual, itd, pad, res, fd, id)

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

        sc.Δt = time() - sc.start_time
        sc.tired = sc.Δt > sc.max_time

        if display == true
            @info log_row(Any[cnts.k, itd.pri_obj, itd.pdd, res.rbNorm, res.rcNorm, res.n_Δx, α_pri, α_dual, itd.μ, itd.regu.ρ, itd.regu.δ])
        end
    end
end
