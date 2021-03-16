export solve_PC!

abstract type DescentDirectionAllocs{T<:Real} end

mutable struct DescentDirectionAllocsPC{T<:Real} <: DescentDirectionAllocs{T}
    Δxy_aff          :: Vector{T} # affine-step solution of the augmented system
    Δs_l_aff         :: Vector{T}
    Δs_u_aff         :: Vector{T} 
    x_m_l_αΔ_aff     :: Vector{T} # x + α * Δxy_aff - lvar
    u_m_x_αΔ_aff     :: Vector{T} # uvar - (x + α * Δxy_aff)
    s_l_αΔ_aff       :: Vector{T} # s_l + α * Δs_l_aff
    s_u_αΔ_aff       :: Vector{T} # s_u + α * Δs_u_aff
    rxs_l            :: Vector{T} # - σ * μ * e + ΔX_aff * Δ_S_l_aff
    rxs_u            :: Vector{T} # σ * μ * e + ΔX_aff * Δ_S_u_aff
end

DescentDirectionAllocsPC(id :: QM_IntData, T :: DataType) = 
    DescentDirectionAllocsPC{T}(zeros(T, id.n_cols+id.n_rows), # Δxy_aff
                                zeros(T, id.n_low), # Δs_l_aff
                                zeros(T, id.n_upp), # Δs_u_aff
                                zeros(T, id.n_low), # x_m_l_αΔ_aff
                                zeros(T, id.n_upp), # u_m_x_αΔ_aff
                                zeros(T, id.n_low), # s_l_αΔ_aff
                                zeros(T, id.n_upp), # s_u_αΔ_aff
                                zeros(T, id.n_low), # rxs_l
                                zeros(T, id.n_upp)  # rxs_u
                                )

convert(::Type{<:DescentDirectionAllocs{T}}, dda :: DescentDirectionAllocsPC{T0}) where {T<:Real, T0<:Real} = 
    DescentDirectionAllocsPC(convert(Array{T}, dda.Δxy_aff),
                             convert(Array{T}, dda.Δs_l_aff),
                             convert(Array{T}, dda.Δs_u_aff),
                             convert(Array{T}, dda.x_m_l_αΔ_aff),
                             convert(Array{T}, dda.u_m_x_αΔ_aff),
                             convert(Array{T}, dda.s_l_αΔ_aff),
                             convert(Array{T}, dda.s_u_αΔ_aff),
                             convert(Array{T}, dda.rxs_l),
                             convert(Array{T}, dda.rxs_u))

function update_pt_aff!(x_m_l_αΔ_aff, u_m_x_αΔ_aff, s_l_αΔ_aff, s_u_αΔ_aff, Δxy_aff, Δs_l_aff, Δs_u_aff, x_m_lvar, uvar_m_x, 
                        s_l, s_u, α_aff_pri, α_aff_dual, ilow, iupp)

    x_m_l_αΔ_aff .= @views x_m_lvar .+ α_aff_pri .* Δxy_aff[ilow]
    u_m_x_αΔ_aff .= @views uvar_m_x .- α_aff_pri .* Δxy_aff[iupp]
    s_l_αΔ_aff .= s_l .+ α_aff_dual .* Δs_l_aff
    s_u_αΔ_aff .= s_u .+ α_aff_dual .* Δs_u_aff
end

# Mehrotra's Predictor-Corrector algorithm
function update_dda!(pt :: Point{T}, itd :: IterData{T}, fd :: Abstract_QM_FloatData{T}, id :: QM_IntData, res :: Residuals{T}, 
                     dda :: DescentDirectionAllocsPC{T}, pad :: PreallocatedData{T}, cnts :: Counters, T0 :: DataType) where {T<:Real} 

    # solve system aff
    dda.Δxy_aff[1:id.n_cols] .= .- res.rc
    dda.Δxy_aff[id.n_cols+1:end] .= .-res.rb
    dda.Δxy_aff[id.ilow] .+= pt.s_l
    dda.Δxy_aff[id.iupp] .-= pt.s_u
    out = solver!(pt, itd, fd, id, res, dda, pad, cnts, T0, :aff)
    out == 1 && return out
    dda.Δs_l_aff .= @views .-pt.s_l .- pt.s_l .* dda.Δxy_aff[id.ilow] ./ itd.x_m_lvar
    dda.Δs_u_aff .= @views .-pt.s_u .+ pt.s_u .* dda.Δxy_aff[id.iupp] ./ itd.uvar_m_x

    α_aff_pri, α_aff_dual = compute_αs(pt.x, pt.s_l, pt.s_u, fd.lvar, fd.uvar, dda.Δxy_aff, dda.Δs_l_aff, 
                                       dda.Δs_u_aff, id.n_cols)

    # (x-lvar, uvar-x, s_l, s_u) .+= α_aff * Δ_aff                                 
    update_pt_aff!(dda.x_m_l_αΔ_aff, dda.u_m_x_αΔ_aff, dda.s_l_αΔ_aff, dda.s_u_αΔ_aff, dda.Δxy_aff, 
                   dda.Δs_l_aff, dda.Δs_u_aff, itd.x_m_lvar, itd.uvar_m_x, pt.s_l, pt.s_u, 
                   α_aff_pri, α_aff_dual, id.ilow, id.iupp)

    μ_aff = compute_μ(dda.x_m_l_αΔ_aff, dda.u_m_x_αΔ_aff, dda.s_l_αΔ_aff, dda.s_u_αΔ_aff, id.n_low, id.n_upp)
    σ = (μ_aff / itd.μ)^3

    # corrector-centering step
    dda.rxs_l .= @views -σ * itd.μ .+ dda.Δxy_aff[id.ilow] .* dda.Δs_l_aff
    dda.rxs_u .= @views σ * itd.μ .+ dda.Δxy_aff[id.iupp] .* dda.Δs_u_aff
    itd.Δxy .= 0
    itd.Δxy[id.ilow] .+= dda.rxs_l ./ itd.x_m_lvar
    itd.Δxy[id.iupp] .+= dda.rxs_u ./ itd.uvar_m_x
    out = solver!(pt, itd, fd, id, res, dda, pad, cnts, T0, :cc)
    out == 1 && return out
    itd.Δs_l .= @views .-(dda.rxs_l .+ pt.s_l .* itd.Δxy[id.ilow]) ./ itd.x_m_lvar
    itd.Δs_u .= @views (dda.rxs_u .+ pt.s_u .* itd.Δxy[id.iupp]) ./ itd.uvar_m_x

    # final direction
    itd.Δxy .+= dda.Δxy_aff  
    itd.Δs_l .+= dda.Δs_l_aff
    itd.Δs_u .+= dda.Δs_u_aff

    return out
end