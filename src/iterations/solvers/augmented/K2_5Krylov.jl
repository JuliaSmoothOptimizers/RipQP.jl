export K2_5KrylovParams

"""
Type to use the K2.5 formulation with a Krylov method, using the package 
[`Krylov.jl`](https://github.com/JuliaSmoothOptimizers/Krylov.jl). 
The outer constructor 

    K2_5KrylovParams(; kmethod = :minres, preconditioner = :Jacobi, atol = 1.0e-10, rtol = 1.0e-10)

creates a [`RipQP.SolverParams`](@ref) that should be used to create a [`RipQP.InputConfig`](@ref).
The available methods are:
- `:minres`
- `:minres_qlp`

The list of available preconditioners for this solver is displayed here: [`RipQP.PreconditionerDataK2`](@ref)
"""
struct K2_5KrylovParams <: SolverParams
  kmethod::Symbol
  preconditioner::Symbol
  atol0::Float64
  rtol0::Float64
  atol_min::Float64
  rtol_min::Float64
  ρ_min::Float64
  δ_min::Float64
end

function K2_5KrylovParams(;
  kmethod::Symbol = :minres,
  preconditioner::Symbol = :Jacobi,
  atol0::T = 1.0e-4,
  rtol0::T = 1.0e-4,
  atol_min::T = 1.0e-10,
  rtol_min::T = 1.0e-10,
  ρ_min::T = 1e2 * sqrt(eps()),
  δ_min::T = 1e3 * sqrt(eps()),
) where {T <: Real}
  return K2_5KrylovParams(kmethod, preconditioner, atol0, rtol0, atol_min, rtol_min, ρ_min, δ_min)
end

mutable struct PreallocatedDataK2_5Krylov{
  T <: Real,
  S,
  L <: LinearOperator,
  Pr <: PreconditionerDataK2,
  Ksol <: KrylovSolver,
} <: PreallocatedDataAugmentedKrylov{T, S}
  pdat::Pr
  D::S                                  # temporary top-left diagonal
  sqrtX1X2::S # vector to scale K2 to K2.5
  tmp::S # temporary vector for products
  rhs::S
  regu::Regularization{T}
  δv::Vector{T}
  K::L # augmented matrix          
  KS::Ksol
  atol::T
  rtol::T
  atol_min::T
  rtol_min::T
end

function opK2_5prod!(
  res::AbstractVector{T},
  nvar::Int,
  Q::AbstractMatrix{T},
  D::AbstractVector{T},
  A::AbstractMatrix{T},
  sqrtX1X2::AbstractVector{T},
  tmp::AbstractVector{T},
  δv::AbstractVector{T},
  v::AbstractVector{T},
  α::T,
  β::T,
  uplo::Symbol,
) where {T}
  @views mul!(tmp, Q, sqrtX1X2 .* v[1:nvar], -α, zero(T))
  tmp .= sqrtX1X2 .* tmp .+ α .* D .* v[1:nvar]
  if β == zero(T)
    res[1:nvar] .= tmp
  else
    res[1:nvar] .= @views tmp .+ β .* res[1:nvar]
  end
  if uplo == :U
    @views mul!(tmp, A, v[(nvar + 1):end], α, zero(T))
    res[1:nvar] .+= sqrtX1X2 .* tmp
    @views mul!(res[(nvar + 1):end], A', sqrtX1X2 .* v[1:nvar], α, β)
  else
    @views mul!(tmp, A', v[(nvar + 1):end], α, zero(T))
    res[1:nvar] .+= sqrtX1X2 .* tmp
    @views mul!(res[(nvar + 1):end], A, sqrtX1X2 .* v[1:nvar], α, β)
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
    regu =
      Regularization(T(sqrt(eps()) * 1e5), T(sqrt(eps()) * 1e5), T(sp.ρ_min), T(sp.δ_min), :classic)
    D .= -T(1.0e0) / 2
  else
    regu = Regularization(
      T(sqrt(eps()) * 1e5),
      T(sqrt(eps()) * 1e5),
      T(sqrt(eps(T)) * 1e0),
      T(sqrt(eps(T)) * 1e0),
      :classic,
    )
    D .= -T(1.0e-2)
  end
  sqrtX1X2 = fill!(similar(D), one(T))
  tmp = similar(D)
  δv = [regu.δ] # put it in a Vector so that we can modify it without modifying opK2prod!
  K = LinearOperator(
    T,
    id.nvar + id.ncon,
    id.nvar + id.ncon,
    true,
    true,
    (res, v, α, β) -> opK2_5prod!(
      res,
      id.nvar,
      Symmetric(fd.Q, fd.uplo),
      D,
      fd.A,
      sqrtX1X2,
      tmp,
      δv,
      v,
      α,
      β,
      fd.uplo,
    ),
  )

  rhs = similar(fd.c, id.nvar + id.ncon)
  kstring = string(sp.kmethod)
  KS = eval(KSolver(sp.kmethod))(K, rhs)

  pdat = eval(sp.preconditioner)(id, fd, regu, D, K)

  return PreallocatedDataK2_5Krylov(
    pdat,
    D,
    sqrtX1X2,
    tmp,
    rhs,
    regu,
    δv,
    K, #K
    KS, #K_fact
    sp.atol0,
    sp.rtol0,
    sp.atol_min,
    sp.rtol_min,
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
  T0::DataType,
  step::Symbol,
) where {T <: Real}

  # erase dda.Δxy_aff only for affine predictor step with PC method
  pad.rhs[1:(id.nvar)] .= @views dd[1:(id.nvar)] .* pad.sqrtX1X2
  pad.rhs[(id.nvar + 1):end] .= @views dd[(id.nvar + 1):end]
  rhsNorm = norm(pad.rhs)
  if rhsNorm != zero(T)
    pad.rhs ./= rhsNorm
  end
  pad.K.nprod = 0
  ksolve!(pad.KS, pad.K, pad.rhs, pad.pdat.P, verbose = 0, atol = pad.atol, rtol = pad.rtol)
  if typeof(res) <: ResidualsHistory
    mul!(res.KΔxy, pad.K, pad.KS.x) # krylov residuals
    res.Kres .= res.KΔxy .- pad.rhs
  end
  if rhsNorm != zero(T)
    pad.KS.x .*= rhsNorm
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

  # K2.5
  pad.sqrtX1X2 .= one(T)
  pad.sqrtX1X2[id.ilow] .*= sqrt.(itd.x_m_lvar)
  pad.sqrtX1X2[id.iupp] .*= sqrt.(itd.uvar_m_x)
  pad.D .= zero(T)
  pad.D[id.ilow] .-= pt.s_l
  pad.D[id.iupp] .*= itd.uvar_m_x
  pad.tmp .= zero(T)
  pad.tmp[id.iupp] .-= pt.s_u
  pad.tmp[id.ilow] .*= itd.x_m_lvar
  pad.D .+= pad.tmp .- pad.regu.ρ

  pad.δv[1] = pad.regu.δ

  update_preconditioner!(pad.pdat, pad, itd, pt, id, fd, cnts)

  return 0
end
