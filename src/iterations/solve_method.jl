export update_dd!, DescentDirectionAllocs, DescentDirectionAllocsPC, DescentDirectionAllocsIPF

"""
Abstract type that wraps data relative to the method used to solve the problem,
such as the Predictor-Corrector Algorithm.
"""
abstract type DescentDirectionAllocs{T<:Real} end

"""
Type that wraps data relative to the Predictor-Corrector algorithm.
To use the Predictor-Corrector algorithm, you should set `solve_method` to `:PC` in the [`RipQP.InputConfig`](@ref) type.

The attributes are :

- `Δxy_aff` : [Δxaff; Δyaff] 
- `Δs_l_aff`
- `Δs_u_aff`
- `x_m_l_αΔ_aff` : x + αaff * Δxaff - lvar
- `u_m_x_αΔ_aff` : uvar - (x + αaff * Δxaff)
- `s_l_αΔ_aff` : sl + αaff * Δslaff
- `s_u_αΔ_aff` : u + αaff * Δsuaff
- `rxs_l` : - σ * μ * e + ΔXaff * ΔSlaff
- `rxs_u` : σ * μ * e + ΔXaff * ΔSuaff

"""
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
function update_dd!(pt :: Point{T}, itd :: IterData{T}, fd :: Abstract_QM_FloatData{T}, id :: QM_IntData, res :: Residuals{T}, 
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

mutable struct DescentDirectionAllocsIPF{T<:Real} <: DescentDirectionAllocs{T}
    check_type :: T # no additional vector required, we use this field only for compat with other methods
end

DescentDirectionAllocsIPF(id :: QM_IntData, T :: DataType) = DescentDirectionAllocsIPF{T}(one(T))

convert(::Type{<:DescentDirectionAllocs{T}}, dda :: DescentDirectionAllocsIPF{T0}) where {T<:Real, T0<:Real} = 
    DescentDirectionAllocsIPF(T(dda.check_type))

function update_dd!(pt :: Point{T}, itd :: IterData{T}, fd :: Abstract_QM_FloatData{T}, id :: QM_IntData, res :: Residuals{T}, 
                    dda :: DescentDirectionAllocsIPF{T}, pad :: PreallocatedData{T}, cnts :: Counters, T0 :: DataType) where {T<:Real} 

    r, γ =  T(0.999), T(0.05)
    # D = [s_l (x-lvar) + s_u (uvar-x)]
    pad.D .= 0
    pad.D[id.ilow] .+= pt.s_l .* itd.x_m_lvar 
    pad.D[id.iupp] .+= pt.s_u .* itd.uvar_m_x 
    pad.D[id.ifree] .= one(T)
    ξ = minimum(pad.D) / itd.μ
    σ = γ * min((one(T) - r) * (one(T) - ξ) / ξ, T(2))^3 

    # rhs = [sqrt(x1x2) (-rc + s_l - s_u) - σμ x1^{-1/2} sqrt(x2) +	σμ sqrt(x1) x2^{-1/2} ]
    #       [                                   -rb                                       ]
    itd.Δxy[1:id.n_cols] .= .-res.rc
    itd.Δxy[id.n_cols+1:end] .= .-res.rb
    itd.Δxy[id.ilow] .+= pt.s_l - σ * itd.μ ./ itd.x_m_lvar
    itd.Δxy[id.iupp] .-= pt.s_u - σ * itd.μ ./ itd.uvar_m_x
    out = solver!(pt, itd, fd, id, res, dda, pad, cnts, T0, :IPF)
    out == 1 && return out
    itd.Δs_l .= @views (σ * itd.μ .- pt.s_l .* itd.Δxy[id.ilow]) ./ itd.x_m_lvar .- pt.s_l 
    itd.Δs_u .= @views (σ * itd.μ .+ pt.s_u .* itd.Δxy[id.iupp]) ./ itd.uvar_m_x .- pt.s_u
end