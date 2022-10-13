export K2_5KrylovParams

"""
Type to use the K2.5 formulation with a Krylov method, using the package 
[`Krylov.jl`](https://github.com/JuliaSmoothOptimizers/Krylov.jl). 
The outer constructor 

    K2_5KrylovParams(; uplo = :L, kmethod = :minres, preconditioner = Identity(),
                     rhs_scale = true,
                     atol0 = 1.0e-4, rtol0 = 1.0e-4, 
                     atol_min = 1.0e-10, rtol_min = 1.0e-10,
                     ρ0 = sqrt(eps()) * 1e5, δ0 = sqrt(eps()) * 1e5, 
                     ρ_min = 1e2 * sqrt(eps()), δ_min = 1e2 * sqrt(eps()),
                     itmax = 0, mem = 20)

creates a [`RipQP.SolverParams`](@ref).
The available methods are:
- `:minres`
- `:minres_qlp`
- `:symmlq`
"""
mutable struct K2_5KrylovParams{T, PT} <: AugmentedKrylovParams{T, PT}
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

function K2_5KrylovParams{T}(;
  uplo::Symbol = :L,
  kmethod::Symbol = :minres,
  preconditioner::AbstractPreconditioner = Identity(),
  rhs_scale::Bool = true,
  atol0::T = 1.0e-4,
  rtol0::T = 1.0e-4,
  atol_min::T = 1.0e-10,
  rtol_min::T = 1.0e-10,
  ρ0::T = sqrt(eps()) * 1e5,
  δ0::T = sqrt(eps()) * 1e5,
  ρ_min::T = 1e2 * sqrt(eps()),
  δ_min::T = 1e3 * sqrt(eps()),
  itmax::Int = 0,
  mem::Int = 20,
) where {T <: Real}
  return K2_5KrylovParams(
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

K2_5KrylovParams(; kwargs...) = K2_5KrylovParams{Float64}(; kwargs...)

mutable struct PreallocatedDataK2_5Krylov{
  T <: Real,
  S,
  L <: LinearOperator,
  Pr <: PreconditionerData,
  Ksol <: KrylovSolver,
} <: PreallocatedDataAugmentedKrylov{T, S}
  pdat::Pr
  D::S                                  # temporary top-left diagonal
  sqrtX1X2::S # vector to scale K2 to K2.5
  tmp1::S # temporary vector for products
  tmp2::S # temporary vector for products
  rhs::S
  rhs_scale::Bool
  regu::Regularization{T}
  δv::Vector{T}
  K::L # augmented matrix          
  KS::Ksol
  kiter::Int
  atol::T
  rtol::T
  atol_min::T
  rtol_min::T
  itmax::Int
end

function opK2_5prod!(
  res::AbstractVector{T},
  nvar::Int,
  Q::Union{AbstractMatrix{T}, AbstractLinearOperator{T}},
  D::AbstractVector{T},
  A::Union{AbstractMatrix{T}, AbstractLinearOperator{T}},
  sqrtX1X2::AbstractVector{T},
  tmp1::AbstractVector{T},
  tmp2::AbstractVector{T},
  δv::AbstractVector{T},
  v::AbstractVector{T},
  α::T,
  β::T,
  uplo::Symbol,
) where {T}
  @. tmp2 = @views sqrtX1X2 * v[1:nvar]
  mul!(tmp1, Q, tmp2, -α, zero(T))
  @. tmp1 = @views sqrtX1X2 * tmp1 + α * D * v[1:nvar]
  if β == zero(T)
    res[1:nvar] .= tmp1
  else
    @. res[1:nvar] = @views tmp1 + β * res[1:nvar]
  end
  if uplo == :U
    @views mul!(tmp1, A, v[(nvar + 1):end], α, zero(T))
    @. res[1:nvar] += sqrtX1X2 * tmp1
    @views mul!(res[(nvar + 1):end], A', tmp2, α, β)
  else
    @views mul!(tmp1, A', v[(nvar + 1):end], α, zero(T))
    @. res[1:nvar] += sqrtX1X2 * tmp1
    @views mul!(res[(nvar + 1):end], A, tmp2, α, β)
  end
  res[(nvar + 1):end] .+= @views (α * δv[1]) .* v[(nvar + 1):end]
end

function PreallocatedData(
  sp::K2_5KrylovParams,
  fd::QM_FloatData{T},
  id::QM_IntData,
  itd::IterData{T},
  pt::Point{T},
  iconf::InputConfig{Tconf},
) where {T <: Real, Tconf <: Real}

  # init Regularization values
  D = similar(fd.c, id.nvar)
  if iconf.mode == :mono
    regu = Regularization(T(sp.ρ0), T(sp.δ0), T(sp.ρ_min), T(sp.δ_min), :classic)
    D .= -T(1.0e0) / 2
  else
    regu =
      Regularization(T(sp.ρ0), T(sp.δ0), T(sqrt(eps(T)) * 1e0), T(sqrt(eps(T)) * 1e0), :classic)
    D .= -T(1.0e-2)
  end
  sqrtX1X2 = fill!(similar(D), one(T))
  tmp1 = similar(D)
  tmp2 = similar(D)
  δv = [regu.δ] # put it in a Vector so that we can modify it without modifying opK2prod!
  K = LinearOperator(
    T,
    id.nvar + id.ncon,
    id.nvar + id.ncon,
    true,
    true,
    (res, v, α, β) ->
      opK2_5prod!(res, id.nvar, fd.Q, D, fd.A, sqrtX1X2, tmp1, tmp2, δv, v, α, β, fd.uplo),
  )

  rhs = similar(fd.c, id.nvar + id.ncon)

  KS = init_Ksolver(K, rhs, sp)

  pdat = PreconditionerData(sp, id, fd, regu, D, K)

  return PreallocatedDataK2_5Krylov(
    pdat,
    D,
    sqrtX1X2,
    tmp1,
    tmp2,
    rhs,
    sp.rhs_scale,
    regu,
    δv,
    K, #K
    KS, #K_fact
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
  pad::PreallocatedDataK2_5Krylov{T},
  dda::DescentDirectionAllocs{T},
  pt::Point{T},
  itd::IterData{T},
  fd::Abstract_QM_FloatData{T},
  id::QM_IntData,
  res::AbstractResiduals{T},
  cnts::Counters,
  step::Symbol,
) where {T <: Real}

  # erase dda.Δxy_aff only for affine predictor step with PC method
  @. pad.rhs[1:(id.nvar)] = @views dd[1:(id.nvar)] * pad.sqrtX1X2
  pad.rhs[(id.nvar + 1):end] .= @views dd[(id.nvar + 1):end]
  if pad.rhs_scale
    rhsNorm = kscale!(pad.rhs)
  end
  (step !== :cc) && (pad.kiter = 0)
  ksolve!(
    pad.KS,
    pad.K,
    pad.rhs,
    pad.pdat.P,
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
  pad.KS.x[1:(id.nvar)] .*= pad.sqrtX1X2

  dd .= pad.KS.x

  return 0
end

function update_pad!(
  pad::PreallocatedDataK2_5Krylov{T},
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

  if pad.atol > pad.atol_min
    pad.atol /= 10
  end
  if pad.rtol > pad.rtol_min
    pad.rtol /= 10
  end

  # K2.5
  pad.sqrtX1X2 .= one(T)
  @. pad.sqrtX1X2[id.ilow] *= sqrt(itd.x_m_lvar)
  @. pad.sqrtX1X2[id.iupp] *= sqrt(itd.uvar_m_x)
  pad.D .= zero(T)
  pad.D[id.ilow] .-= pt.s_l
  pad.D[id.iupp] .*= itd.uvar_m_x
  pad.tmp1 .= zero(T)
  pad.tmp1[id.iupp] .-= pt.s_u
  pad.tmp1[id.ilow] .*= itd.x_m_lvar
  @. pad.D += pad.tmp1 - pad.regu.ρ

  pad.δv[1] = pad.regu.δ

  update_preconditioner!(pad.pdat, pad, itd, pt, id, fd, cnts)

  return 0
end
