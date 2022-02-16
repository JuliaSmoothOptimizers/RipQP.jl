export K1_2StructuredParams

"""
Type to use the K1.2 formulation with a structured Krylov method, using the package 
[`Krylov.jl`](https://github.com/JuliaSmoothOptimizers/Krylov.jl).
This only works for solving Linear Problems.
The outer constructor 

    K1_2StructuredParams(; uplo = :L, kmethod = :craig, atol0 = 1.0e-4, rtol0 = 1.0e-4,
                         atol_min = 1.0e-10, rtol_min = 1.0e-10, 
                         ρ_min = 1e3 * sqrt(eps()), δ_min = 1e4 * sqrt(eps()),
                         mem = 20)

creates a [`RipQP.SolverParams`](@ref) that should be used to create a [`RipQP.InputConfig`](@ref).
The available methods are:
- `:lnlq`
- `:craig`
- `:craigmr`

"""
mutable struct K1_2StructuredParams <: NormalParams
  uplo::Symbol
  kmethod::Symbol
  atol0::Float64
  rtol0::Float64
  atol_min::Float64
  rtol_min::Float64
  ρ_min::Float64
  δ_min::Float64
  mem::Int
end

function K1_2StructuredParams(;
  uplo::Symbol = :L,
  kmethod::Symbol = :craig,
  atol0::T = 1.0e-4,
  rtol0::T = 1.0e-4,
  atol_min::T = 1.0e-10,
  rtol_min::T = 1.0e-10,
  ρ_min::T = 1e3 * sqrt(eps()),
  δ_min::T = 1e4 * sqrt(eps()),
  mem::Int = 20,
) where {T <: Real}
  return K1_2StructuredParams(uplo, kmethod, atol0, rtol0, atol_min, rtol_min, ρ_min, δ_min, mem)
end

mutable struct PreallocatedDataK1_2Structured{T <: Real, S, Ksol <: KrylovSolver} <:
               PreallocatedDataNormalStructured{T, S}
  E::S  # temporary top-left diagonal
  invE::S
  ξ1::S
  ξ2::S # todel
  Δx0::S 
  ξ22::S # todel
  regu::Regularization{T}
  KS::Ksol
  atol::T
  rtol::T
  atol_min::T
  rtol_min::T
end

function PreallocatedData(
  sp::K1_2StructuredParams,
  fd::QM_FloatData{T},
  id::QM_IntData,
  itd::IterData{T},
  pt::Point{T},
  iconf::InputConfig{Tconf},
) where {T <: Real, Tconf <: Real}

  @assert iconf.solve_method == :IPF
  
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
    regu,
    KS,
    T(sp.atol0),
    T(sp.rtol0),
    T(sp.atol_min),
    T(sp.rtol_min),
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
  T0::DataType,
  step::Symbol,
) where {T <: Real}

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

  # rhsNorm = kscale!(pad.rhs)
  # pad.K.nprod = 0
  ksolve!(
    pad.KS,
    fd.uplo == :U ? fd.A' : fd.A,
    pad.ξ22,
    Diagonal(pad.invE),
    pad.regu.δ,
    verbose = 0,
    atol = pad.atol,
    rtol = pad.rtol,
  )
  dd[(id.nvar + 1):end] .= pad.KS.y
  dd[1:(id.nvar)] .= pad.KS.x .- pad.Δx0
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
