export K1_1StructuredParams

"""
Type to use the K1.1 formulation with a structured Krylov method, using the package 
[`Krylov.jl`](https://github.com/JuliaSmoothOptimizers/Krylov.jl).
This only works for solving Linear Problems.
The outer constructor 

    K1_1StructuredParams(; uplo = :L, kmethod = :lsqr, rhs_scale = true,
                         atol0 = 1.0e-4, rtol0 = 1.0e-4,
                         atol_min = 1.0e-10, rtol_min = 1.0e-10, 
                         ρ_min = 1e3 * sqrt(eps()), δ_min = 1e4 * sqrt(eps()),
                         itmax = 0, mem = 20)

creates a [`RipQP.SolverParams`](@ref).
The available methods are:
- `:lslq`
- `:lsqr`
- `:lsmr`

"""
mutable struct K1_1StructuredParams <: NormalParams
  uplo::Symbol
  kmethod::Symbol
  rhs_scale::Bool
  atol0::Float64
  rtol0::Float64
  atol_min::Float64
  rtol_min::Float64
  ρ_min::Float64
  δ_min::Float64
  itmax::Int
  mem::Int
end

function K1_1StructuredParams(;
  uplo::Symbol = :L,
  kmethod::Symbol = :lsqr,
  rhs_scale::Bool = true,
  atol0::T = 1.0e-4,
  rtol0::T = 1.0e-4,
  atol_min::T = 1.0e-10,
  rtol_min::T = 1.0e-10,
  ρ_min::T = 1e3 * sqrt(eps()),
  δ_min::T = 1e4 * sqrt(eps()),
  itmax::Int = 0,
  mem::Int = 20,
) where {T <: Real}
  return K1_1StructuredParams(
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

mutable struct PreallocatedDataK1_1Structured{T <: Real, S, Ksol <: KrylovSolver} <:
               PreallocatedDataNormalKrylovStructured{T, S}
  E::S  # temporary top-left diagonal
  invE::S
  ξ1::S
  ξ2::S # todel
  Δy0::S
  ξ12::S # todel
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
  sp::K1_1StructuredParams,
  fd::QM_FloatData{T},
  id::QM_IntData,
  itd::IterData{T},
  pt::Point{T},
  iconf::InputConfig{Tconf},
) where {T <: Real, Tconf <: Real}
  @assert typeof(iconf.solve_method) <: IPF

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
  Δy0 = similar(fd.c, id.ncon)
  ξ12 = similar(fd.c, id.nvar)

  KS = init_Ksolver(fd.uplo == :U ? fd.A : fd.A', ξ12, sp)

  return PreallocatedDataK1_1Structured(
    E,
    invE,
    ξ1,
    ξ2,
    Δy0,
    ξ12,
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
  pad::PreallocatedDataK1_1Structured{T},
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
  pad.ξ1 .= @views step == :init ? fd.c : dd[1:(id.nvar)]
  pad.ξ2 .= @views (step == :init && all(dd[(id.nvar + 1):end] .== zero(T))) ? one(T) :
         dd[(id.nvar + 1):end]

  if pad.regu.δ == zero(T)
    pad.Δy0 .= zero(T)
  else
    pad.Δy0 .= .-pad.ξ2 ./ pad.regu.δ
  end
  if fd.uplo == :U
    mul!(pad.ξ12, fd.A, pad.Δy0)
  else
    mul!(pad.ξ12, fd.A', pad.Δy0)
  end
  pad.ξ12 .+= pad.ξ1
  if pad.rhs_scale
    ξ12Norm = kscale!(pad.ξ12)
  end
  (step !== :cc) && (pad.kiter = 0)
  ksolve!(
    pad.KS,
    fd.uplo == :U ? fd.A : fd.A',
    pad.ξ12,
    Diagonal(pad.invE),
    pad.regu.δ,
    verbose = 0,
    atol = pad.atol,
    rtol = pad.rtol,
    itmax = pad.itmax,
  )
  if pad.rhs_scale
    kunscale!(pad.KS.x, ξ12Norm)
  end
  dd[(id.nvar + 1):end] .= pad.KS.x .- pad.Δy0
  if fd.uplo == :U
    @views mul!(pad.ξ1, fd.A, dd[(id.nvar + 1):end], one(T), -one(T))
  else
    @views mul!(pad.ξ1, fd.A', dd[(id.nvar + 1):end], one(T), -one(T))
  end
  dd[1:(id.nvar)] .= pad.ξ1 ./ pad.E
  update_kresiduals_history_K1struct!(
    res,
    fd.uplo == :U ? fd.A' : fd.A,
    pad.E,
    pad.invE, # tmp storage vector
    pad.regu.δ,
    pad.KS.x,
    pad.ξ12,
    :K1_1,
  )
  pad.kiter += niterations(pad.KS)

  return 0
end

function update_pad!(
  pad::PreallocatedDataK1_1Structured{T},
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
  pad.invE .= one(T) ./ pad.E

  return 0
end
