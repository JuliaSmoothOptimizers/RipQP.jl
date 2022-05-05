export PC, IPF

mutable struct PC <: SolveMethod end

abstract type DescentDirectionAllocs{T <: Real, S} end

mutable struct DescentDirectionAllocsPC{T <: Real, S} <: DescentDirectionAllocs{T, S}
  Δxy_aff::S # affine-step solution of the augmented system [Δx_aff; Δy_aff], size nvar + ncon 
  Δs_l_aff::S # size nlow
  Δs_u_aff::S # size nupp
  x_m_l_αΔ_aff::S # x + α_aff * Δxy_aff - lvar , size nlow
  u_m_x_αΔ_aff::S # uvar - (x + α_aff * Δxy_aff) , size nupp
  s_l_αΔ_aff::S # s_l + α_aff * Δs_l_aff , size nlow
  s_u_αΔ_aff::S # s_u + α_aff * Δs_u_aff , size nupp
  rxs_l::S # - σ * μ * e + ΔX_aff * Δ_S_l_aff , size nlow
  rxs_u::S # σ * μ * e + ΔX_aff * Δ_S_u_aff , size nupp
  function DescentDirectionAllocsPC(
    Δxy_aff::AbstractVector{T},
    Δs_l_aff::AbstractVector{T},
    Δs_u_aff::AbstractVector{T},
    x_m_l_αΔ_aff::AbstractVector{T},
    u_m_x_αΔ_aff::AbstractVector{T},
    s_l_αΔ_aff::AbstractVector{T},
    s_u_αΔ_aff::AbstractVector{T},
    rxs_l::AbstractVector{T},
    rxs_u::AbstractVector{T},
  ) where {T <: Real}
    S = typeof(Δxy_aff)
    return new{T, S}(
      Δxy_aff,
      Δs_l_aff,
      Δs_u_aff,
      x_m_l_αΔ_aff,
      u_m_x_αΔ_aff,
      s_l_αΔ_aff,
      s_u_αΔ_aff,
      rxs_l,
      rxs_u,
    )
  end
end

DescentDirectionAllocs(id::QM_IntData, sm::PC, S::DataType) where {T <: Real} =
  DescentDirectionAllocsPC(
    S(undef, id.nvar + id.ncon), # Δxy_aff
    S(undef, id.nlow), # Δs_l_aff
    S(undef, id.nupp), # Δs_u_aff
    S(undef, id.nlow), # x_m_l_αΔ_aff
    S(undef, id.nupp), # u_m_x_αΔ_aff
    S(undef, id.nlow), # s_l_αΔ_aff
    S(undef, id.nupp), # s_u_αΔ_aff
    S(undef, id.nlow), # rxs_l
    S(undef, id.nupp),  # rxs_u
  )

convert(
  ::Type{<:DescentDirectionAllocs{T, S}},
  dda::DescentDirectionAllocsPC{T0, S0},
) where {T <: Real, S <: AbstractVector{T}, T0 <: Real, S0} = DescentDirectionAllocsPC(
  convert(S, dda.Δxy_aff),
  convert(S, dda.Δs_l_aff),
  convert(S, dda.Δs_u_aff),
  convert(S, dda.x_m_l_αΔ_aff),
  convert(S, dda.u_m_x_αΔ_aff),
  convert(S, dda.s_l_αΔ_aff),
  convert(S, dda.s_u_αΔ_aff),
  convert(S, dda.rxs_l),
  convert(S, dda.rxs_u),
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
  res::AbstractResiduals{T},
  pad::PreallocatedData{T},
  cnts::Counters,
  T0::DataType,
) where {T <: Real}

  # solve system aff
  dda.Δxy_aff[1:(id.nvar)] .= .-res.rc
  dda.Δxy_aff[(id.nvar + 1):(id.nvar + id.ncon)] .= .-res.rb
  if typeof(pad) <: PreallocatedDataAugmented || typeof(pad) <: PreallocatedDataNormal
    dda.Δxy_aff[id.ilow] .+= pt.s_l
    dda.Δxy_aff[id.iupp] .-= pt.s_u
  elseif typeof(pad) <: PreallocatedDataNewton
    dda.Δs_l_aff .= .-itd.x_m_lvar .* pt.s_l
    dda.Δs_u_aff .= .-itd.uvar_m_x .* pt.s_u
  end

  cnts.w.write == true && write_system(cnts.w, pad.K, dda.Δxy_aff, :aff, cnts.k)
  out = @timeit_debug to "solver aff" solver!(
    dda.Δxy_aff,
    pad,
    dda,
    pt,
    itd,
    fd,
    id,
    res,
    cnts,
    T0,
    :aff,
  )
  out == 1 && return out
  if typeof(pad) <: PreallocatedDataAugmented || typeof(pad) <: PreallocatedDataNormal
    dda.Δs_l_aff .= @views .-pt.s_l .- pt.s_l .* dda.Δxy_aff[id.ilow] ./ itd.x_m_lvar
    dda.Δs_u_aff .= @views .-pt.s_u .+ pt.s_u .* dda.Δxy_aff[id.iupp] ./ itd.uvar_m_x
  end

  if typeof(pt.x) <: Vector
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
  else
    α_aff_pri, α_aff_dual = compute_αs_gpu(
      pt.x,
      pt.s_l,
      pt.s_u,
      fd.lvar,
      fd.uvar,
      dda.Δxy_aff,
      dda.Δs_l_aff,
      dda.Δs_u_aff,
      id.nvar,
      itd.store_vpri,
      itd.store_vdual_l,
      itd.store_vdual_u,
    )
  end

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
  if typeof(pad) <: PreallocatedDataAugmented || typeof(pad) <: PreallocatedDataNormal
    dda.rxs_l .= @views -σ * itd.μ .+ dda.Δxy_aff[id.ilow] .* dda.Δs_l_aff
    dda.rxs_u .= @views σ * itd.μ .+ dda.Δxy_aff[id.iupp] .* dda.Δs_u_aff
    itd.Δxy .= 0
    itd.Δxy[id.ilow] .+= dda.rxs_l ./ itd.x_m_lvar
    itd.Δxy[id.iupp] .+= dda.rxs_u ./ itd.uvar_m_x
  elseif typeof(pad) <: PreallocatedDataNewton
    itd.Δxy[1:end] .= 0
    itd.Δs_l .= @views σ * itd.μ .- dda.Δxy_aff[id.ilow] .* dda.Δs_l_aff
    itd.Δs_u .= @views σ * itd.μ .+ dda.Δxy_aff[id.iupp] .* dda.Δs_u_aff
  end

  cnts.w.write == true && write_system(cnts.w, pad.K, itd.Δxy, :cc, cnts.k)
  out = @timeit_debug to "solver cc" solver!(itd.Δxy, pad, dda, pt, itd, fd, id, res, cnts, T0, :cc)
  out == 1 && return out
  if typeof(pad) <: PreallocatedDataAugmented || typeof(pad) <: PreallocatedDataNormal
    itd.Δs_l .= @views .-(dda.rxs_l .+ pt.s_l .* itd.Δxy[id.ilow]) ./ itd.x_m_lvar
    itd.Δs_u .= @views (dda.rxs_u .+ pt.s_u .* itd.Δxy[id.iupp]) ./ itd.uvar_m_x
  end

  # final direction
  itd.Δxy .+= dda.Δxy_aff
  itd.Δs_l .+= dda.Δs_l_aff
  itd.Δs_u .+= dda.Δs_u_aff

  return out
end

mutable struct IPF <: SolveMethod
  r::Float64
  γ::Float64
end

IPF(; r::Float64 = 0.999, γ::Float64 = 0.05) = IPF(r, γ)

mutable struct DescentDirectionAllocsIPF{T <: Real, S} <: DescentDirectionAllocs{T, S}
  r::T
  γ::T
  compl_l::S # complementarity s_lᵀ(x-lvar)
  compl_u::S # complementarity s_uᵀ(uvar-x)
  function DescentDirectionAllocsIPF(
    r::T,
    γ::T,
    compl_l::AbstractVector{T},
    compl_u::AbstractVector{T},
  ) where {T <: Real}
    S = typeof(compl_l)
    return new{T, S}(r, γ, compl_l, compl_u)
  end
end

function DescentDirectionAllocs(id::QM_IntData, sm::IPF, S::DataType)
  T = eltype(S)
  return DescentDirectionAllocsIPF(T(sm.r), T(sm.γ), S(undef, id.nlow), S(undef, id.nupp))
end

convert(
  ::Type{<:DescentDirectionAllocs{T, S}},
  dda::DescentDirectionAllocsIPF{T0, S0},
) where {T <: Real, S <: AbstractVector{T}, T0 <: Real, S0} =
  DescentDirectionAllocsIPF(T(dda.r), T(dda.γ), convert(S, dda.compl_l), convert(S, dda.compl_u))

function update_dd!(
  dda::DescentDirectionAllocsIPF{T},
  pt::Point{T},
  itd::IterData{T},
  fd::Abstract_QM_FloatData{T},
  id::QM_IntData,
  res::AbstractResiduals{T},
  pad::PreallocatedData{T},
  cnts::Counters,
  T0::DataType,
) where {T <: Real}
  # D = [s_l (x-lvar) + s_u (uvar-x)]
  dda.compl_l .= pt.s_l .* itd.x_m_lvar
  dda.compl_u .= pt.s_u .* itd.uvar_m_x
  min_compl_l = (id.nlow > 0) ? minimum(dda.compl_l) / (sum(dda.compl_l) / id.nlow) : one(T)
  min_compl_u = (id.nupp > 0) ? minimum(dda.compl_u) / (sum(dda.compl_u) / id.nupp) : one(T)
  ξ = min(min_compl_l, min_compl_u)
  σ = dda.γ * min((one(T) - dda.r) * (one(T) - ξ) / ξ, T(2))^3

  itd.Δxy[1:(id.nvar)] .= .-res.rc
  itd.Δxy[(id.nvar + 1):(id.nvar + id.ncon)] .= .-res.rb
  if typeof(pad) <: PreallocatedDataAugmented || typeof(pad) <: PreallocatedDataNormal
    itd.Δxy[id.ilow] .+= pt.s_l - σ * itd.μ ./ itd.x_m_lvar
    itd.Δxy[id.iupp] .-= pt.s_u - σ * itd.μ ./ itd.uvar_m_x
  elseif typeof(pad) <: PreallocatedDataNewton
    itd.Δs_l .= σ * itd.μ .- itd.x_m_lvar .* pt.s_l
    itd.Δs_u .= σ * itd.μ .- itd.uvar_m_x .* pt.s_u
  end

  cnts.w.write == true && write_system(cnts.w, pad.K, itd.Δxy, :IPF, cnts.k)
  out =
    @timeit_debug to "solver IPF" solver!(itd.Δxy, pad, dda, pt, itd, fd, id, res, cnts, T0, :IPF)
  out == 1 && return out
  if typeof(pad) <: PreallocatedDataAugmented || typeof(pad) <: PreallocatedDataNormal
    itd.Δs_l .= @views (σ * itd.μ .- pt.s_l .* itd.Δxy[id.ilow]) ./ itd.x_m_lvar .- pt.s_l
    itd.Δs_u .= @views (σ * itd.μ .+ pt.s_u .* itd.Δxy[id.iupp]) ./ itd.uvar_m_x .- pt.s_u
  end
end
