export solve_PC!

# Mehrotra's Predictor-Corrector algorithm
function solve_PC!(pt :: point{T}, itd :: iter_data{T}, fd :: Abstract_QM_FloatData{T}, id :: QM_IntData, res :: residuals{T}, 
                   pad :: preallocated_data_K2{T}, cnts :: counters, T0 :: DataType) where {T<:Real} 

    # solve system aff
    pad.Δxy_aff[1:id.n_cols] .= .- res.rc
    pad.Δxy_aff[id.n_cols+1:end] .= .-res.rb
    pad.Δxy_aff[id.ilow] .+= pt.s_l
    pad.Δxy_aff[id.iupp] .-= pt.s_u
    out = solver!(pt, itd, fd, id, res, pad, cnts, T0, :aff)
    out == 1 && return out
    pad.Δs_l_aff .= @views .-pt.s_l .- pt.s_l .* pad.Δxy_aff[id.ilow] ./ itd.x_m_lvar
    pad.Δs_u_aff .= @views .-pt.s_u .+ pt.s_u .* pad.Δxy_aff[id.iupp] ./ itd.uvar_m_x

    α_aff_pri, α_aff_dual = compute_αs(pt.x, pt.s_l, pt.s_u, fd.lvar, fd.uvar, pad.Δxy_aff, pad.Δs_l_aff, 
                                       pad.Δs_u_aff, id.n_cols)

    # (x-lvar, uvar-x, s_l, s_u) .+= α_aff * Δ_aff                                 
    update_pt_aff!(pad.x_m_l_αΔ_aff, pad.u_m_x_αΔ_aff, pad.s_l_αΔ_aff, pad.s_u_αΔ_aff, pad.Δxy_aff, 
                   pad.Δs_l_aff, pad.Δs_u_aff, itd.x_m_lvar, itd.uvar_m_x, pt.s_l, pt.s_u, 
                   α_aff_pri, α_aff_dual, id.ilow, id.iupp)

    μ_aff = compute_μ(pad.x_m_l_αΔ_aff, pad.u_m_x_αΔ_aff, pad.s_l_αΔ_aff, pad.s_u_αΔ_aff, id.n_low, id.n_upp)
    σ = (μ_aff / itd.μ)^3

    # corrector-centering step
    pad.rxs_l .= @views -σ * itd.μ .+ pad.Δxy_aff[id.ilow] .* pad.Δs_l_aff
    pad.rxs_u .= @views σ * itd.μ .+ pad.Δxy_aff[id.iupp] .* pad.Δs_u_aff
    pad.Δxy .= 0
    pad.Δxy[id.ilow] .+= pad.rxs_l ./ itd.x_m_lvar
    pad.Δxy[id.iupp] .+= pad.rxs_u ./ itd.uvar_m_x
    out = solver!(pt, itd, fd, id, res, pad, cnts, T0, :cc)
    out == 1 && return out
    pad.Δs_l .= @views .-(pad.rxs_l .+ pt.s_l .* pad.Δxy[id.ilow]) ./ itd.x_m_lvar
    pad.Δs_u .= @views (pad.rxs_u .+ pt.s_u .* pad.Δxy[id.iupp]) ./ itd.uvar_m_x

    # final direction
    pad.Δxy .+= pad.Δxy_aff  
    pad.Δs_l .+= pad.Δs_l_aff
    pad.Δs_u .+= pad.Δs_u_aff

    return out
end