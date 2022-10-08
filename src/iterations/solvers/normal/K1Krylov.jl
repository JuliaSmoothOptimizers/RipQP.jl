# (A D⁻¹ Aᵀ + δI) Δy = A D⁻¹ ξ₁ + ξ₂ 
# where D = s_l (x - lvar)⁻¹ + s_u (uvar - x)⁻¹ + ρI,
# and the right hand side of K2 is rhs = [ξ₁]
#                                        [ξ₂] 
export K1KrylovParams

"""
Type to use the K1 formulation with a Krylov method, using the package 
[`Krylov.jl`](https://github.com/JuliaSmoothOptimizers/Krylov.jl). 
The outer constructor 

    K1KrylovParams(; uplo = :L, kmethod = :cg, preconditioner = Identity(),
                   rhs_scale = true,
                   atol0 = 1.0e-4, rtol0 = 1.0e-4, 
                   atol_min = 1.0e-10, rtol_min = 1.0e-10,
                   ρ0 = sqrt(eps()) * 1e5, δ0 = sqrt(eps()) * 1e5, 
                   ρ_min = 1e2 * sqrt(eps()), δ_min = 1e2 * sqrt(eps()),
                   itmax = 0, mem = 20)

creates a [`RipQP.SolverParams`](@ref).
The available methods are:
- `:cg`
- `:cg_lanczos`
- `:cr`
- `:minres`
- `:minres_qlp`
- `:symmlq`

"""
mutable struct K1KrylovParams{T, PT} <: NormalKrylovParams{T, PT}
  uplo::Symbol
  kmethod::Symbol
  preconditioner::PT
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

function K1KrylovParams{T}(;
  uplo::Symbol = :L,
  kmethod::Symbol = :cg,
  preconditioner::AbstractPreconditioner = Identity(),
  rhs_scale::Bool = true,
  atol0::T = eps(T)^(1 / 4),
  rtol0::T = eps(T)^(1 / 4),
  atol_min::T = sqrt(eps(T)),
  rtol_min::T = sqrt(eps(T)),
  ρ0::T = T(sqrt(eps()) * 1e5),
  δ0::T = T(sqrt(eps()) * 1e5),
  ρ_min::T = T(1e3 * sqrt(eps())),
  δ_min::T = T(1e4 * sqrt(eps())),
  itmax::Int = 0,
  mem::Int = 20,
) where {T <: Real}
  return K1KrylovParams(
    uplo,
    kmethod,
    preconditioner,
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

K1KrylovParams(; kwargs...) = K1KrylovParams{Float64}(; kwargs...)

mutable struct PreallocatedDataK1Krylov{T <: Real, S, L <: LinearOperator, Ksol <: KrylovSolver} <:
               PreallocatedDataNormalKrylov{T, S}
  D::S
  rhs::S
  rhs_scale::Bool
  regu::Regularization{T}
  δv::Vector{T}
  K::L # augmented matrix (LinearOperator)         
  KS::Ksol
  kiter::Int
  atol::T
  rtol::T
  atol_min::T
  rtol_min::T
  itmax::Int
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
  KS = init_Ksolver(K, rhs, sp)

  return PreallocatedDataK1Krylov(
    D,
    rhs,
    sp.rhs_scale,
    regu,
    δv,
    K, #K
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
  pad::PreallocatedDataK1Krylov{T},
  dda::DescentDirectionAllocs{T},
  pt::Point{T},
  itd::IterData{T},
  fd::Abstract_QM_FloatData{T},
  id::QM_IntData,
  res::AbstractResiduals{T},
  cnts::Counters,
  ::Type{T0},
  step::Symbol,
) where {T <: Real, T0 <: Real}
  pad.rhs .= @views dd[(id.nvar + 1):end]
  if fd.uplo == :U
    @views mul!(pad.rhs, fd.A', dd[1:(id.nvar)] ./ pad.D, one(T), one(T))
  else
    @views mul!(pad.rhs, fd.A, dd[1:(id.nvar)] ./ pad.D, one(T), one(T))
  end
  if pad.rhs_scale
    rhsNorm = kscale!(pad.rhs)
  end
  (step !== :cc) && (pad.kiter = 0)
  ksolve!(
    pad.KS,
    pad.K,
    pad.rhs,
    I,
    verbose = 0,
    atol = pad.atol,
    rtol = pad.rtol,
    itmax = pad.itmax,
  )
  update_kresiduals_history!(res, pad.K, pad.KS.x, pad.rhs)
  pad.kiter += niterations(pad.KS)
  if pad.rhs_scale
    kunscale!(pad.KS.x, rhsNorm)
  end

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
  ::Type{T0},
) where {T <: Real, T0 <: Real}
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
