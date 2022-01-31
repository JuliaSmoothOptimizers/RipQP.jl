export K2_5StructuredParams

"""
Type to use the K2.5 formulation with a structured Krylov method, using the package 
[`Krylov.jl`](https://github.com/JuliaSmoothOptimizers/Krylov.jl).
This only works for solving Linear Problems.
The outer constructor 

    K2_5StructuredParams(; uplo = :L, kmethod = :trimr, atol0 = 1.0e-4, rtol0 = 1.0e-4,
                         atol_min = 1.0e-10, rtol_min = 1.0e-10,
                         ρ0 = sqrt(eps()) * 1e5, δ0 = sqrt(eps()) * 1e5,
                         ρ_min = 1e2 * sqrt(eps()), δ_min = 1e2 * sqrt(eps()))

creates a [`RipQP.SolverParams`](@ref) that should be used to create a [`RipQP.InputConfig`](@ref).
The available methods are:
- `:tricg`
- `:trimr`
- `:gpmr`

The list of available preconditioners for this solver is displayed here: [`RipQP.PreconditionerDataK2`](@ref).
The `mem` argument sould be used only with `gpmr`.
"""
mutable struct K2_5StructuredParams <: AugmentedParams
  uplo::Symbol
  kmethod::Symbol
  atol0::Float64
  rtol0::Float64
  atol_min::Float64
  rtol_min::Float64
  ρ0::Float64
  δ0::Float64
  ρ_min::Float64
  δ_min::Float64
  mem::Int
end

function K2_5StructuredParams(;
  uplo::Symbol = :L,
  kmethod::Symbol = :trimr,
  atol0::T = 1.0e-4,
  rtol0::T = 1.0e-4,
  atol_min::T = 1.0e-10,
  rtol_min::T = 1.0e-10,
  ρ0::T = sqrt(eps()) * 1e5,
  δ0::T = sqrt(eps()) * 1e5,
  ρ_min::T = 1e2 * sqrt(eps()),
  δ_min::T = 1e2 * sqrt(eps()),
  mem::Int = 20,
) where {T <: Real}
  return K2_5StructuredParams(
    uplo,
    kmethod,
    atol0,
    rtol0,
    atol_min,
    rtol_min,
    ρ0,
    δ0,
    ρ_min,
    δ_min,
    mem,
  )
end

mutable struct PreallocatedDataK2_5Structured{
  T <: Real,
  S,
  Ksol <: KrylovSolver,
  L <: AbstractLinearOperator{T},
} <: PreallocatedDataAugmentedStructured{T, S}
  E::S                                  # temporary top-left diagonal
  sqrtX1X2::S # vector to scale K2 to K2.5
  AsqrtX1X2::L
  ξ1::S
  ξ2::S
  regu::Regularization{T}
  KS::Ksol
  atol::T
  rtol::T
  atol_min::T
  rtol_min::T
end

get_nprod!(pad::PreallocatedDataK2_5Structured) = 0

function PreallocatedData(
  sp::K2_5StructuredParams,
  fd::QM_FloatData{T},
  id::QM_IntData,
  itd::IterData{T},
  pt::Point{T},
  iconf::InputConfig{Tconf},
) where {T <: Real, Tconf <: Real}

  # init Regularization values
  E = similar(fd.c, id.nvar)
  if iconf.mode == :mono
    regu = Regularization(T(sp.ρ0), T(sp.δ0), T(sp.ρ_min), T(sp.δ_min), :classic)
    E .= T(1.0e0) / 2
  else
    regu =
      Regularization(T(sp.ρ0), T(sp.δ0), T(sqrt(eps(T)) * 1e0), T(sqrt(eps(T)) * 1e0), :classic)
    E .= T(1.0e-2)
  end
  if regu.δ_min == zero(T) # gsp for gpmr
    regu.δ = zero(T)
  end

  sqrtX1X2 = fill!(similar(fd.c), one(T))
  ξ1 = similar(fd.c, id.nvar)
  ξ2 = similar(fd.c, id.ncon)
  if sp.kmethod == :gpmr
    KS = eval(KSolver(sp.kmethod))(fd.A', fd.b, sp.mem)
  else
    KS = eval(KSolver(sp.kmethod))(fd.A', fd.b)
  end
  AsqrtX1X2 = LinearOperator(fd.A) * Diagonal(sqrtX1X2)

  return PreallocatedDataK2_5Structured(
    E,
    sqrtX1X2,
    AsqrtX1X2,
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

function solver!(
  dd::AbstractVector{T},
  pad::PreallocatedDataK2_5Structured{T},
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
  pad.ξ1 .= @views step == :init ? fd.c : dd[1:(id.nvar)] .* pad.sqrtX1X2
  pad.ξ2 .= @views (step == :init && all(dd[(id.nvar + 1):end] .== zero(T))) ? one(T) :
         dd[(id.nvar + 1):end]
  # rhsNorm = kscale!(pad.rhs)
  # pad.K.nprod = 0
  ksolve!(
    pad.KS,
    pad.AsqrtX1X2',
    pad.ξ1,
    pad.ξ2,
    inv(Diagonal(pad.E)),
    (one(T) / pad.regu.δ) .* I,
    verbose = 0,
    atol = pad.atol,
    rtol = pad.rtol,
    gsp = (pad.regu.δ == zero(T)),
  )
  update_kresiduals_history!(
    res,
    pad.E,
    fd.A,
    pad.regu.δ,
    pad.KS.x,
    pad.KS.y,
    pad.ξ1,
    pad.ξ2,
    id.nvar,
  )
  # kunscale!(pad.KS.x, rhsNorm)

  dd[1:(id.nvar)] .= pad.KS.x .* pad.sqrtX1X2
  dd[(id.nvar + 1):end] .= pad.KS.y

  return 0
end

function update_pad!(
  pad::PreallocatedDataK2_5Structured{T},
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

  pad.sqrtX1X2 .= one(T)
  pad.sqrtX1X2[id.ilow] .*= sqrt.(itd.x_m_lvar)
  pad.sqrtX1X2[id.iupp] .*= sqrt.(itd.uvar_m_x)
  pad.E .= pad.regu.ρ
  pad.E[id.ilow] .+= pt.s_l ./ itd.x_m_lvar
  pad.E[id.iupp] .+= pt.s_u ./ itd.uvar_m_x
  pad.E .*= pad.sqrtX1X2 .^ 2

  return 0
end
