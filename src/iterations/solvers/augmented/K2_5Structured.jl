export K2_5StructuredParams

"""
Type to use the K2.5 formulation with a structured Krylov method, using the package 
[`Krylov.jl`](https://github.com/JuliaSmoothOptimizers/Krylov.jl).
This only works for solving Linear Problems.
The outer constructor 

    K2_5StructuredParams(; uplo = :L, kmethod = :trimr, rhs_scale = true,
                         atol0 = 1.0e-4, rtol0 = 1.0e-4,
                         atol_min = 1.0e-10, rtol_min = 1.0e-10,
                         ρ0 = sqrt(eps()) * 1e5, δ0 = sqrt(eps()) * 1e5,
                         ρ_min = 1e2 * sqrt(eps()), δ_min = 1e2 * sqrt(eps()),
                         itmax = 0, mem = 20)

creates a [`RipQP.SolverParams`](@ref).
The available methods are:
- `:tricg`
- `:trimr`
- `:gpmr`

The `mem` argument sould be used only with `gpmr`.
"""
mutable struct K2_5StructuredParams{T} <: AugmentedParams{T}
  uplo::Symbol
  kmethod::Symbol
  rhs_scale::Bool
  atol0::T
  rtol0::T
  atol_min::T
  rtol_min::T
  ρ0::T
  δ0::T
  ρ_min::T
  δ_min::T
  itmax::Int
  mem::Int
end

function K2_5StructuredParams{T}(;
  uplo::Symbol = :L,
  kmethod::Symbol = :trimr,
  rhs_scale::Bool = true,
  atol0::T = eps(T)^(1/4),
  rtol0::T = eps(T)^(1/4),
  atol_min::T = sqrt(eps(T)),
  rtol_min::T = sqrt(eps(T)),
  ρ0::T = eps(T)^(1/4),
  δ0::T = eps(T)^(1/4),
  ρ_min::T = T(1e2 * sqrt(eps(T))),
  δ_min::T = T(1e2 * sqrt(eps(T))),
  itmax::Int = 0,
  mem::Int = 20,
) where {T <: Real}
  return K2_5StructuredParams(
    uplo,
    kmethod,
    rhs_scale,
    atol0,
    rtol0,
    atol_min,
    rtol_min,
    ρ0,
    δ0,
    ρ_min,
    δ_min,
    itmax,
    mem,
  )
end

K2_5StructuredParams(; kwargs...) = K2_5StructuredParams{Float64}(; kwargs...)

mutable struct PreallocatedDataK2_5Structured{
  T <: Real,
  S,
  Ksol <: KrylovSolver,
  L <: AbstractLinearOperator{T},
} <: PreallocatedDataAugmentedKrylovStructured{T, S}
  E::S                                  # temporary top-left diagonal
  invE::S
  sqrtX1X2::S # vector to scale K2 to K2.5
  AsqrtX1X2::L
  ξ1::S
  ξ2::S
  rhs_scale::Bool
  regu::Regularization{T}
  KS::Ksol
  kiter::Int
  atol::T
  rtol::T
  atol_min::T
  rtol_min::T
  itmax::Int
end

function opAsqrtX1X2tprod!(res, A, v, α, β, sqrtX1X2)
  mul!(res, transpose(A), v, α, β)
  res .*= sqrtX1X2
end

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
  invE = similar(E)
  invE .= one(T) ./ E

  sqrtX1X2 = fill!(similar(fd.c), one(T))
  ξ1 = similar(fd.c, id.nvar)
  ξ2 = similar(fd.c, id.ncon)

  KS = init_Ksolver(fd.A', fd.b, sp)

  AsqrtX1X2 = LinearOperator(
    T,
    id.ncon,
    id.nvar,
    false,
    false,
    (res, v, α, β) -> mul!(res, fd.A, v .* sqrtX1X2, α, β),
    (res, v, α, β) -> opAsqrtX1X2tprod!(res, fd.A, v, α, β, sqrtX1X2),
  )

  return PreallocatedDataK2_5Structured(
    E,
    invE,
    sqrtX1X2,
    AsqrtX1X2,
    ξ1,
    ξ2,
    sp.rhs_scale,
    regu,
    KS,
    0,
    T(sp.atol0),
    T(sp.rtol0),
    T(sp.atol_min),
    T(sp.rtol_min),
    sp.itmax,
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
  if pad.rhs_scale
    rhsNorm = sqrt(norm(pad.ξ1)^2 + norm(pad.ξ2)^2)
    pad.ξ1 ./= rhsNorm
    pad.ξ2 ./= rhsNorm
  end
  (step !== :cc) && (pad.kiter = 0)
  ksolve!(
    pad.KS,
    pad.AsqrtX1X2',
    pad.ξ1,
    pad.ξ2,
    Diagonal(pad.invE),
    (one(T) / pad.regu.δ) .* I,
    verbose = 0,
    atol = pad.atol,
    rtol = pad.rtol,
    gsp = (pad.regu.δ == zero(T)),
    itmax = pad.itmax,
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
  pad.kiter += niterations(pad.KS)
  if pad.rhs_scale
    kunscale!(pad.KS.x, rhsNorm)
    kunscale!(pad.KS.y, rhsNorm)
  end

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
  pad.invE .= one(T) ./ pad.E

  return 0
end
