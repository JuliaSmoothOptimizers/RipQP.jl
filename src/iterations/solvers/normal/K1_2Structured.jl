export K1_2StructuredParams

"""
Type to use the K1.2 formulation with a structured Krylov method, using the package 
[`Krylov.jl`](https://github.com/JuliaSmoothOptimizers/Krylov.jl).
This only works for solving Linear Problems.
The outer constructor 

    K1_2StructuredParams(; uplo = :L, kmethod = :craig, rhs_scale = true,
                         atol0 = 1.0e-4, rtol0 = 1.0e-4,
                         atol_min = 1.0e-10, rtol_min = 1.0e-10, 
                         ρ_min = 1e3 * sqrt(eps()), δ_min = 1e4 * sqrt(eps()),
                         itmax = 0, mem = 20)

creates a [`RipQP.SolverParams`](@ref).
The available methods are:
- `:lnlq`
- `:craig`
- `:craigmr`

"""
mutable struct K1_2StructuredParams{T} <: NormalParams{T}
  uplo::Symbol
  kmethod::Symbol
  rhs_scale::Bool
  atol0::T
  rtol0::T
  atol_min::T
  rtol_min::T
  ρ_min::T
  δ_min::T
  itmax::Int
  mem::Int
end

function K1_2StructuredParams{T}(;
  uplo::Symbol = :L,
  kmethod::Symbol = :craig,
  rhs_scale::Bool = true,
  atol0::T = eps(T)^(1 / 4),
  rtol0::T = eps(T)^(1 / 4),
  atol_min::T = sqrt(eps(T)),
  rtol_min::T = sqrt(eps(T)),
  ρ_min::T = T(1e3 * sqrt(eps())),
  δ_min::T = T(1e4 * sqrt(eps())),
  itmax::Int = 0,
  mem::Int = 20,
) where {T <: Real}
  return K1_2StructuredParams(
    uplo,
    kmethod,
    rhs_scale,
    atol0,
    rtol0,
    atol_min,
    rtol_min,
    ρ_min,
    δ_min,
    itmax,
    mem,
  )
end

K1_2StructuredParams(; kwargs...) = K1_2StructuredParams{Float64}(; kwargs...)

mutable struct PreallocatedDataK1_2Structured{T <: Real, S, Ksol <: KrylovSolver} <:
               PreallocatedDataNormalKrylovStructured{T, S}
  E::S  # temporary top-left diagonal
  invE::S
  ξ1::S
  ξ2::S # todel
  Δx0::S
  ξ22::S # todel
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

function PreallocatedData(
  sp::K1_2StructuredParams,
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
  if regu.δ_min == zero(T) # gsp for gpmr
    regu.δ = zero(T)
  end
  invE = similar(E)
  invE .= one(T) ./ E

  ξ1 = similar(fd.c, id.nvar)
  ξ2 = similar(fd.c, id.ncon)
  Δx0 = similar(fd.c, id.nvar)
  ξ22 = similar(fd.c, id.ncon)

  KS = init_Ksolver(fd.uplo == :U ? fd.A' : fd.A, ξ22, sp)

  return PreallocatedDataK1_2Structured(
    E,
    invE,
    ξ1,
    ξ2,
    Δx0,
    ξ22,
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
  pad::PreallocatedDataK1_2Structured{T},
  dda::DescentDirectionAllocs{T},
  pt::Point{T},
  itd::IterData{T},
  fd::Abstract_QM_FloatData{T},
  id::QM_IntData,
  res::AbstractResiduals{T},
  cnts::Counters,
  step::Symbol,
) where {T <: Real}
  @assert typeof(dda) <: DescentDirectionAllocsIPF
  pad.ξ1 .= @views step == :init ? fd.c : dd[1:(id.nvar)]
  pad.ξ2 .= @views (step == :init && all(dd[(id.nvar + 1):end] .== zero(T))) ? one(T) :
         dd[(id.nvar + 1):end]

  pad.Δx0 .= pad.ξ1 ./ pad.E
  if fd.uplo == :U
    mul!(pad.ξ22, fd.A', pad.Δx0)
  else
    mul!(pad.ξ22, fd.A, pad.Δx0)
  end
  pad.ξ22 .+= pad.ξ2
  if pad.rhs_scale
    ξ22Norm = kscale!(pad.ξ22)
  end
  (step !== :cc) && (pad.kiter = 0)
  ksolve!(
    pad.KS,
    fd.uplo == :U ? fd.A' : fd.A,
    pad.ξ22,
    Diagonal(pad.invE),
    pad.regu.δ,
    verbose = 0,
    atol = pad.atol,
    rtol = pad.rtol,
    itmax = pad.itmax,
  )
  if pad.rhs_scale
    kunscale!(pad.KS.x, ξ22Norm)
    kunscale!(pad.KS.y, ξ22Norm)
  end
  dd[(id.nvar + 1):end] .= pad.KS.y
  @. dd[1:(id.nvar)] = pad.KS.x - pad.Δx0
  update_kresiduals_history_K1struct!(
    res,
    fd.uplo == :U ? fd.A' : fd.A,
    pad.E,
    pad.invE, # tmp storage vector
    pad.regu.δ,
    pad.KS.y,
    pad.ξ22,
    :K1_2,
  )
  # kunscale!(pad.KS.x, rhsNorm)
  pad.kiter += niterations(pad.KS)

  return 0
end

function update_pad!(
  pad::PreallocatedDataK1_2Structured{T},
  dda::DescentDirectionAllocs{T},
  pt::Point{T},
  itd::IterData{T},
  fd::Abstract_QM_FloatData{T},
  id::QM_IntData,
  res::AbstractResiduals{T},
  cnts::Counters,
) where {T <: Real}
  if cnts.k != 0
    update_regu!(pad.regu)
  end

  update_krylov_tol!(pad)

  pad.E .= pad.regu.ρ
  @. pad.E[id.ilow] += pt.s_l / itd.x_m_lvar
  @. pad.E[id.iupp] += pt.s_u / itd.uvar_m_x
  @. pad.invE = one(T) / pad.E

  return 0
end
