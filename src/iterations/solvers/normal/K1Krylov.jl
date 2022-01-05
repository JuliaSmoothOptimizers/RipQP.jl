# (A D⁻¹ Aᵀ + δI) Δy = A D⁻¹ ξ₁ + ξ₂ 
# where D = s_l (x - lvar)⁻¹ + s_u (uvar - x)⁻¹ + ρI,
# and the right hand side of K2 is rhs = [ξ₁]
#                                        [ξ₂] 
export K1KrylovParams

"""
Type to use the K1 formulation with a Krylov method, using the package 
[`Krylov.jl`](https://github.com/JuliaSmoothOptimizers/Krylov.jl). 
The outer constructor 

    K1KrylovParams(; uplo = :L, kmethod = :cg, preconditioner = :Identity,
                   atol0 = 1.0e-4, rtol0 = 1.0e-4, 
                   atol_min = 1.0e-10, rtol_min = 1.0e-10,
                   ρ0 = sqrt(eps()) * 1e5, δ0 = sqrt(eps()) * 1e5, 
                   ρ_min = 1e2 * sqrt(eps()), δ_min = 1e2 * sqrt(eps()))

creates a [`RipQP.SolverParams`](@ref) that should be used to create a [`RipQP.InputConfig`](@ref).
The available methods are:
- `:cg`
- `:minres`
- `:minres_qlp`

"""
mutable struct K1KrylovParams <: SolverParams
  uplo::Symbol
  kmethod::Symbol
  preconditioner::Symbol
  atol0::Float64
  rtol0::Float64
  atol_min::Float64
  rtol_min::Float64
  ρ0::Float64
  δ0::Float64
  ρ_min::Float64
  δ_min::Float64
end

function K1KrylovParams(;
  uplo::Symbol = :L,
  kmethod::Symbol = :cg,
  preconditioner::Symbol = :Identity,
  atol0::T = 1.0e-4,
  rtol0::T = 1.0e-4,
  atol_min::T = 1.0e-10,
  rtol_min::T = 1.0e-10,
  ρ0::T = sqrt(eps()) * 1e5,
  δ0::T = sqrt(eps()) * 1e5,
  ρ_min::T = 1e3 * sqrt(eps()),
  δ_min::T = 1e4 * sqrt(eps()),
) where {T <: Real}
  return K1KrylovParams(
    uplo,
    kmethod,
    preconditioner,
    atol0,
    rtol0,
    atol_min,
    rtol_min,
    ρ0,
    δ0,
    ρ_min,
    δ_min,
  )
end

mutable struct PreallocatedDataK1Krylov{T <: Real, S, L <: LinearOperator, Ksol <: KrylovSolver} <:
               PreallocatedDataNormalKrylov{T, S}
  D::S
  rhs::S
  regu::Regularization{T}
  δv::Vector{T}
  K::L # augmented matrix (LinearOperator)         
  KS::Ksol
  atol::T
  rtol::T
  atol_min::T
  rtol_min::T
end

function opK1prod!(
  res::AbstractVector{T},
  D::AbstractVector{T},
  A::Union{AbstractMatrix{T}, AbstractLinearOperator{T}},
  δv::AbstractVector{T},
  v::AbstractVector{T},
  vtmp::AbstractVector{T},
  α::T,
  β::T,
  uplo::Symbol,
) where {T}
  if uplo == :U
    mul!(vtmp, A, v)
    mul!(res, A', vtmp ./ D, α, β)
    res .+= (α * δv[1]) .* v
  else
    mul!(vtmp, A', v)
    mul!(res, A, vtmp ./ D, α, β)
    res .+= (α * δv[1]) .* v
  end
end

function PreallocatedData(
  sp::K1KrylovParams,
  fd::QM_FloatData{T},
  id::QM_IntData,
  itd::IterData{T},
  pt::Point{T},
  iconf::InputConfig{Tconf},
) where {T <: Real, Tconf <: Real}
  D = similar(fd.c, id.nvar)
  # init Regularization values
  if iconf.mode == :mono
    regu = Regularization(T(sp.ρ0), T(sp.δ0), T(sp.ρ_min), T(sp.δ_min), :classic)
    # Regularization(T(0.), T(0.), T(sp.ρ_min), T(sp.δ_min), :classic)
    D .= T(1.0e0) / 2
  else
    regu =
      Regularization(T(sp.ρ0), T(sp.δ0), T(sqrt(eps(T)) * 1e0), T(sqrt(eps(T)) * 1e0), :classic)
    D .= T(1.0e-2)
  end

  δv = [regu.δ] # put it in a Vector so that we can modify it without modifying opK2prod!
  K = LinearOperator(
    T,
    id.ncon,
    id.ncon,
    true,
    true,
    (res, v, α, β) -> opK1prod!(res, D, fd.A, δv, v, similar(fd.c), α, β, fd.uplo),
  )

  rhs = similar(fd.c, id.ncon)
  kstring = string(sp.kmethod)
  KS = eval(KSolver(sp.kmethod))(K, rhs)

  return PreallocatedDataK1Krylov(
    D,
    rhs,
    regu,
    δv,
    K, #K
    KS,
    sp.atol0,
    sp.rtol0,
    sp.atol_min,
    sp.rtol_min,
  )
end

function solver!(
  dd::AbstractVector{T},
  pad::PreallocatedDataK1Krylov{T},
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
  pad.rhs .= @views dd[(id.nvar + 1):end]
  if fd.uplo == :U
    @views mul!(pad.rhs, fd.A', dd[1:(id.nvar)] ./ pad.D, one(T), one(T))
  else
    @views mul!(pad.rhs, fd.A, dd[1:(id.nvar)] ./ pad.D, one(T), one(T))
  end
  rhsNorm = kscale!(pad.rhs)
  pad.K.nprod = 0
  ksolve!(pad.KS, pad.K, pad.rhs, I(id.ncon), verbose = 0, atol = pad.atol, rtol = pad.rtol)
  update_kresiduals_history!(res, pad.K, pad.KS.x, pad.rhs)
  kunscale!(pad.KS.x, rhsNorm)

  if fd.uplo == :U
    @views mul!(dd[1:(id.nvar)], fd.A, pad.KS.x, one(T), -one(T))
  else
    @views mul!(dd[1:(id.nvar)], fd.A', pad.KS.x, one(T), -one(T))
  end
  dd[1:(id.nvar)] ./= pad.D
  dd[(id.nvar + 1):end] .= pad.KS.x
  return 0
end

function update_pad!(
  pad::PreallocatedDataK1Krylov{T},
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

  pad.δv[1] = pad.regu.δ

  pad.D .= pad.regu.ρ
  pad.D[id.ilow] .+= pt.s_l ./ itd.x_m_lvar
  pad.D[id.iupp] .+= pt.s_u ./ itd.uvar_m_x
  pad.δv[1] = pad.regu.δ

  # update_preconditioner!(pad.pdat, pad, itd, pt, id, fd, cnts)

  return 0
end
