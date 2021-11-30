export K2StructuredParams

"""
Type to use the K2 formulation with a structured Krylov method, using the package 
[`Krylov.jl`](https://github.com/JuliaSmoothOptimizers/Krylov.jl). 
The outer constructor 

    K2StructuredParams(; uplo = :L, kmethod = :trimr, ratol = 1.0e-10, rrtol = 1.0e-10)

creates a [`RipQP.SolverParams`](@ref) that should be used to create a [`RipQP.InputConfig`](@ref).
The available methods are:
- `:tricg`
- `:trimr`

The list of available preconditioners for this solver is displayed here: [`RipQP.PreconditionerDataK2`](@ref).
"""
struct K2StructuredParams <: SolverParams
  uplo::Symbol
  kmethod::Symbol
  atol0::Float64
  rtol0::Float64
  atol_min::Float64
  rtol_min::Float64
  ρ_min::Float64
  δ_min::Float64
end

function K2StructuredParams(;
  uplo::Symbol = :L,
  kmethod::Symbol = :trimr,
  atol0::T = 1.0e-4,
  rtol0::T = 1.0e-4,
  atol_min::T = 1.0e-10,
  rtol_min::T = 1.0e-10,
  ρ_min::T = 1e2 * sqrt(eps()),
  δ_min::T = 1e2 * sqrt(eps()),
) where {T <: Real}
  return K2StructuredParams(
    uplo,
    kmethod,
    atol0,
    rtol0,
    atol_min,
    rtol_min,
    ρ_min,
    δ_min,
  )
end

mutable struct PreallocatedDataK2Structured{
  T <: Real,
  S,
  Ksol <: KrylovSolver,
} <: PreallocatedDataAugmentedStructured{T, S}
  E::S                                  # temporary top-left diagonal
  ξ1::S
  ξ2::S
  regu::Regularization{T}
  KS::Ksol
  atol::T
  rtol::T
  atol_min::T
  rtol_min::T
end

get_nprod!(pad::PreallocatedDataK2Structured) = 0

function PreallocatedData(
  sp::K2StructuredParams,
  fd::QM_FloatData{T},
  id::QM_IntData,
  itd::IterData{T},
  pt::Point{T},
  iconf::InputConfig{Tconf},
) where {T <: Real, Tconf <: Real}

  # init Regularization values
  E = similar(fd.c, id.nvar)
  if iconf.mode == :mono
    regu =
      Regularization(T(sqrt(eps()) * 1e5), T(sqrt(eps()) * 1e5), T(sp.ρ_min), T(sp.δ_min), :classic)
    E .= T(1.0e0) / 2
  else
    regu = Regularization(
      T(sqrt(eps()) * 1e5),
      T(sqrt(eps()) * 1e5),
      T(sqrt(eps(T)) * 1e0),
      T(sqrt(eps(T)) * 1e0),
      :classic,
    )
    E .= T(1.0e-2)
  end

  ξ1 = similar(fd.c, id.nvar)
  ξ2 = similar(fd.c, id.ncon)
  KS = eval(KSolver(sp.kmethod))(fd.A', fd.b)

  return PreallocatedDataK2Structured(
    E,
    ξ1,
    ξ2,
    regu,
    KS,
    sp.atol0,
    sp.rtol0,
    sp.atol_min,
    sp.rtol_min,
  )
end

function update_kresiduals_history!(
  res::AbstractResiduals{T},
  E::AbstractVector{T},
  A::Union{AbstractMatrix{T}, AbstractLinearOperator{T}},
  δ::T,
  solx::AbstractVector{T},
  soly::AbstractVector{T},
  ξ1::AbstractVector{T},
  ξ2::AbstractVector{T},
  nvar::Int,
) where {T <: Real}
  if typeof(res) <: ResidualsHistory
    @views mul!(res.Kres[1:nvar], A', soly)
    res.Kres[1:nvar] .+= .-E .* solx .- ξ1
    @views mul!(res.Kres[nvar+1: end], A, solx)
    res.Kres[nvar+1: end] .+= δ .* soly .- ξ2
  end
end

function solver!(
  dd::AbstractVector{T},
  pad::PreallocatedDataK2Structured{T},
  dda::DescentDirectionAllocs{T},
  pt::Point{T},
  itd::IterData{T},
  fd::Abstract_QM_FloatData{T},
  id::QM_IntData,
  res::AbstractResiduals{T},
  cnts::Counters,
  T0::DataType,
  step::Symbol,
) where {T <: Real}
  pad.ξ1 .= step == :init ? fd.c : dd[1:id.nvar]
  pad.ξ2 .= (step == :init && all(dd[id.nvar+1: end] .== zero(T))) ? one(T) : dd[id.nvar+1: end]
  # rhsNorm = kscale!(pad.rhs)
  # pad.K.nprod = 0
  ksolve!(pad.KS, fd.A', pad.ξ1, pad.ξ2, inv(Diagonal(pad.E)), (one(T)/pad.regu.δ) .* I, verbose = 0, atol = pad.atol, rtol = pad.rtol)
  update_kresiduals_history!(res, pad.E, fd.A, pad.regu.δ, pad.KS.x, pad.KS.y, pad.ξ1, pad.ξ2, id.nvar)
  # kunscale!(pad.KS.x, rhsNorm)

  dd[1:id.nvar] .= pad.KS.x
  dd[id.nvar+1: end] .= pad.KS.y

  return 0
end

function update_pad!(
  pad::PreallocatedDataK2Structured{T},
  dda::DescentDirectionAllocs{T},
  pt::Point{T},
  itd::IterData{T},
  fd::Abstract_QM_FloatData{T},
  id::QM_IntData,
  res::AbstractResiduals{T},
  cnts::Counters,
  T0::DataType,
) where {T <: Real}
  if cnts.k != 0
    update_regu!(pad.regu)
  end

  if pad.atol > pad.atol_min
    pad.atol /= 10
  end
  if pad.rtol > pad.rtol_min
    pad.rtol /= 10
  end

  pad.E .= pad.regu.ρ
  pad.E[id.ilow] .+= pt.s_l ./ itd.x_m_lvar
  pad.E[id.iupp] .+= pt.s_u ./ itd.uvar_m_x

  return 0
end
