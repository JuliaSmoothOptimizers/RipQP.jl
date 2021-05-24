abstract type DescentDirectionAllocs{T <: Real} end

mutable struct DescentDirectionAllocsPC{T <: Real} <: DescentDirectionAllocs{T}
  Δxy_aff::Vector{T} # affine-step solution of the augmented system [Δx_aff; Δy_aff], size nvar + ncon 
  Δs_l_aff::Vector{T} # size nlow
  Δs_u_aff::Vector{T} # size nupp
  x_m_l_αΔ_aff::Vector{T} # x + α_aff * Δxy_aff - lvar , size nlow
  u_m_x_αΔ_aff::Vector{T} # uvar - (x + α_aff * Δxy_aff) , size nupp
  s_l_αΔ_aff::Vector{T} # s_l + α_aff * Δs_l_aff , size nlow
  s_u_αΔ_aff::Vector{T} # s_u + α_aff * Δs_u_aff , size nupp
  rxs_l::Vector{T} # - σ * μ * e + ΔX_aff * Δ_S_l_aff , size nlow
  rxs_u::Vector{T} # σ * μ * e + ΔX_aff * Δ_S_u_aff , size nupp
end

DescentDirectionAllocsPC(id::QM_IntData, fd::QM_FloatData{T}) where {T <: Real} = DescentDirectionAllocsPC{T}(
  similar(fd.c, id.nvar + id.ncon), # Δxy_aff
  similar(fd.c, id.nlow), # Δs_l_aff
  similar(fd.c, id.nupp), # Δs_u_aff
  similar(fd.c, id.nlow), # x_m_l_αΔ_aff
  similar(fd.c, id.nupp), # u_m_x_αΔ_aff
  similar(fd.c, id.nlow), # s_l_αΔ_aff
  similar(fd.c, id.nupp), # s_u_αΔ_aff
  similar(fd.c, id.nlow), # rxs_l
  similar(fd.c, id.nupp),  # rxs_u
)

convert(
  ::Type{<:DescentDirectionAllocs{T}},
  dda::DescentDirectionAllocsPC{T0},
) where {T <: Real, T0 <: Real} = DescentDirectionAllocsPC(
  convert(Array{T}, dda.Δxy_aff),
  convert(Array{T}, dda.Δs_l_aff),
  convert(Array{T}, dda.Δs_u_aff),
  convert(Array{T}, dda.x_m_l_αΔ_aff),
  convert(Array{T}, dda.u_m_x_αΔ_aff),
  convert(Array{T}, dda.s_l_αΔ_aff),
  convert(Array{T}, dda.s_u_αΔ_aff),
  convert(Array{T}, dda.rxs_l),
  convert(Array{T}, dda.rxs_u),
)

function update_pt_aff!(
  x_m_l_αΔ_aff,
  u_m_x_αΔ_aff,
  s_l_αΔ_aff,
  s_u_αΔ_aff,
  Δxy_aff,
  Δs_l_aff,
  Δs_u_aff,
  x_m_lvar,
  uvar_m_x,
  s_l,
  s_u,
  α_aff_pri,
  α_aff_dual,
  ilow,
  iupp,
)
  x_m_l_αΔ_aff .= @views x_m_lvar .+ α_aff_pri .* Δxy_aff[ilow]
  u_m_x_αΔ_aff .= @views uvar_m_x .- α_aff_pri .* Δxy_aff[iupp]
  s_l_αΔ_aff .= s_l .+ α_aff_dual .* Δs_l_aff
  s_u_αΔ_aff .= s_u .+ α_aff_dual .* Δs_u_aff
end

# Mehrotra's Predictor-Corrector algorithm
function update_dd!(
  dda::DescentDirectionAllocsPC{T},
  pt::Point{T},
  itd::IterData{T},
  fd::Abstract_QM_FloatData{T},
  id::QM_IntData,
  res::Residuals{T},
  pad::PreallocatedData{T},
  cnts::Counters,
  T0::DataType,
) where {T <: Real}

  # solve system aff
  dda.Δxy_aff[1:(id.nvar)] .= .-res.rc
  dda.Δxy_aff[(id.nvar + 1):end] .= .-res.rb
  dda.Δxy_aff[id.ilow] .+= pt.s_l
  dda.Δxy_aff[id.iupp] .-= pt.s_u

  cnts.w.write == true && write_system(cnts.w, pad.K, dda.Δxy_aff, :aff, cnts.k)
  out = solver!(pad, dda, pt, itd, fd, id, res, cnts, T0, :aff)
  out == 1 && return out
  dda.Δs_l_aff .= @views .-pt.s_l .- pt.s_l .* dda.Δxy_aff[id.ilow] ./ itd.x_m_lvar
  dda.Δs_u_aff .= @views .-pt.s_u .+ pt.s_u .* dda.Δxy_aff[id.iupp] ./ itd.uvar_m_x

  α_aff_pri, α_aff_dual = compute_αs(
    pt.x,
    pt.s_l,
    pt.s_u,
    fd.lvar,
    fd.uvar,
    dda.Δxy_aff,
    dda.Δs_l_aff,
    dda.Δs_u_aff,
    id.nvar,
  )

  # (x-lvar, uvar-x, s_l, s_u) .+= α_aff * Δ_aff                                 
  update_pt_aff!(
    dda.x_m_l_αΔ_aff,
    dda.u_m_x_αΔ_aff,
    dda.s_l_αΔ_aff,
    dda.s_u_αΔ_aff,
    dda.Δxy_aff,
    dda.Δs_l_aff,
    dda.Δs_u_aff,
    itd.x_m_lvar,
    itd.uvar_m_x,
    pt.s_l,
    pt.s_u,
    α_aff_pri,
    α_aff_dual,
    id.ilow,
    id.iupp,
  )

  μ_aff =
    compute_μ(dda.x_m_l_αΔ_aff, dda.u_m_x_αΔ_aff, dda.s_l_αΔ_aff, dda.s_u_αΔ_aff, id.nlow, id.nupp)
  σ = (μ_aff / itd.μ)^3

  # corrector-centering step
  dda.rxs_l .= @views -σ * itd.μ .+ dda.Δxy_aff[id.ilow] .* dda.Δs_l_aff
  dda.rxs_u .= @views σ * itd.μ .+ dda.Δxy_aff[id.iupp] .* dda.Δs_u_aff
  itd.Δxy .= 0
  itd.Δxy[id.ilow] .+= dda.rxs_l ./ itd.x_m_lvar
  itd.Δxy[id.iupp] .+= dda.rxs_u ./ itd.uvar_m_x

  cnts.w.write == true && write_system(cnts.w, pad.K, itd.Δxy, :cc, cnts.k)
  out = solver!(pad, dda, pt, itd, fd, id, res, cnts, T0, :cc)
  out == 1 && return out
  itd.Δs_l .= @views .-(dda.rxs_l .+ pt.s_l .* itd.Δxy[id.ilow]) ./ itd.x_m_lvar
  itd.Δs_u .= @views (dda.rxs_u .+ pt.s_u .* itd.Δxy[id.iupp]) ./ itd.uvar_m_x

  # final direction
  itd.Δxy .+= dda.Δxy_aff
  itd.Δs_l .+= dda.Δs_l_aff
  itd.Δs_u .+= dda.Δs_u_aff

  return out
end

mutable struct DescentDirectionAllocsIPF{T <: Real} <: DescentDirectionAllocs{T}
  compl::Vector{T} # complementarity s_lᵀ(x-lvar) + s_uᵀ(uvar-x)
end

DescentDirectionAllocsIPF(id::QM_IntData, fd::QM_FloatData{T}) where {T <: Real} =
  DescentDirectionAllocsIPF{T}(similar(fd.c, id.nvar))

convert(
  ::Type{<:DescentDirectionAllocs{T}},
  dda::DescentDirectionAllocsIPF{T0},
) where {T <: Real, T0 <: Real} = DescentDirectionAllocsIPF(convert(Array{T}, dda.compl))

function update_dd!(
  dda::DescentDirectionAllocsIPF{T},
  pt::Point{T},
  itd::IterData{T},
  fd::Abstract_QM_FloatData{T},
  id::QM_IntData,
  res::Residuals{T},
  pad::PreallocatedData{T},
  cnts::Counters,
  T0::DataType,
) where {T <: Real}
  r, γ = T(0.999), T(0.05)
  # D = [s_l (x-lvar) + s_u (uvar-x)]
  dda.compl .= 0
  dda.compl[id.ilow] .+= pt.s_l .* itd.x_m_lvar
  dda.compl[id.iupp] .+= pt.s_u .* itd.uvar_m_x
  dda.compl[id.ifree] .= one(T)
  ξ = minimum(dda.compl) / itd.μ
  σ = γ * min((one(T) - r) * (one(T) - ξ) / ξ, T(2))^3

  itd.Δxy[1:(id.nvar)] .= .-res.rc
  itd.Δxy[(id.nvar + 1):end] .= .-res.rb
  itd.Δxy[id.ilow] .+= pt.s_l - σ * itd.μ ./ itd.x_m_lvar
  itd.Δxy[id.iupp] .-= pt.s_u - σ * itd.μ ./ itd.uvar_m_x

  cnts.w.write == true && write_system(cnts.w, pad.K, itd.Δxy, :IPF, cnts.k)
  out = solver!(pad, dda, pt, itd, fd, id, res, cnts, T0, :IPF)
  out == 1 && return out
  itd.Δs_l .= @views (σ * itd.μ .- pt.s_l .* itd.Δxy[id.ilow]) ./ itd.x_m_lvar .- pt.s_l
  itd.Δs_u .= @views (σ * itd.μ .+ pt.s_u .* itd.Δxy[id.iupp]) ./ itd.uvar_m_x .- pt.s_u
end
