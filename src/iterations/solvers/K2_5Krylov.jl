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
  atol::Float64
  rtol::Float64
  ρ_min::Float64
  δ_min::Float64
end

function K2_5KrylovParams(;
  kmethod::Symbol = :minres,
  preconditioner::Symbol = :Jacobi,
  atol::T = 1.0e-10,
  rtol::T = 1.0e-10,
  ρ_min::T = 1e2 * sqrt(eps()),
  δ_min::T = 1e3 * sqrt(eps()),
) where {T <: Real}
  return K2_5KrylovParams(kmethod, preconditioner, atol, rtol, ρ_min, δ_min)
end

function opK2_5prod!(
  res::AbstractVector{T},
  nvar::Int,
  Q::AbstractMatrix{T},
  D::AbstractVector{T},
  AT::AbstractMatrix{T},
  sqrtX1X2::AbstractVector{T},
  tmp::AbstractVector{T},
  δv::AbstractVector{T},
  v::AbstractVector{T},
  α::T,
  β::T,
) where {T}
  @views mul!(tmp, Q, sqrtX1X2 .* v[1:nvar], -α, zero(T))
  tmp .= sqrtX1X2 .* tmp .+ α .* D .* v[1:nvar]
  if β == zero(T)
    res[1:nvar] .= tmp
  else
    res[1:nvar] .= @views tmp .+ β .* res[1:nvar]
  end
  @views mul!(tmp, AT, v[(nvar + 1):end], α, zero(T))
  res[1:nvar] .+= sqrtX1X2 .* tmp
  @views mul!(res[(nvar + 1):end], AT', sqrtX1X2 .* v[1:nvar], α, β)
  res[(nvar + 1):end] .+= @views (α * δv[1]) .* v[(nvar + 1):end]
end

function PreallocatedData(
  sp::K2_5KrylovParams,
  fd::QM_FloatData{T},
  id::QM_IntData,
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
    (res, v, α, β) -> opK2_5prod!(res, id.nvar, fd.Q, D, fd.AT, sqrtX1X2, tmp, δv, v, α, β),
  )

  rhs = similar(fd.c, id.nvar + id.ncon)
  kstring = string(sp.kmethod)
  KS = eval(KSolver(sp.kmethod))(K, rhs)

  pdat = eval(sp.preconditioner)(id, fd, regu, D, K)

  return eval(Symbol(:PreallocatedData_K2_5, sp.kmethod))(
    pdat,
    D,
    sqrtX1X2,
    tmp,
    rhs,
    regu,
    δv,
    K, #K
    KS, #K_fact
    sp.atol,
    sp.rtol,
  )
end

function solver!(
  pad::PreallocatedData_K2_5Krylov{T},
  dda::DescentDirectionAllocs{T},
  pt::Point{T},
  itd::IterData{T},
  fd::Abstract_QM_FloatData{T},
  id::QM_IntData,
  res::Residuals{T},
  cnts::Counters,
  T0::DataType,
  step::Symbol,
) where {T <: Real}

  # erase dda.Δxy_aff only for affine predictor step with PC method
  if step == :aff
    pad.rhs[1:(id.nvar)] .= @views dda.Δxy_aff[1:(id.nvar)] .* pad.sqrtX1X2
    pad.rhs[(id.nvar + 1):end] .= @views dda.Δxy_aff[(id.nvar + 1):end]
  else
    pad.rhs[1:(id.nvar)] .= @views itd.Δxy[1:(id.nvar)] .* pad.sqrtX1X2
    pad.rhs[(id.nvar + 1):end] .= @views itd.Δxy[(id.nvar + 1):end]
  end
  rhsNorm = norm(pad.rhs)
  if rhsNorm != zero(T)
    pad.rhs ./= rhsNorm
  end
  ksolve!(pad.KS, pad.K, pad.rhs, pad.pdat.P, verbose = 0, atol = pad.atol, rtol = pad.rtol)
  if rhsNorm != zero(T)
    pad.KS.x .*= rhsNorm
  end
  pad.KS.x[1:(id.nvar)] .*= pad.sqrtX1X2

  if step == :aff
    dda.Δxy_aff .= pad.KS.x
  else
    itd.Δxy .= pad.KS.x
  end

  return 0
end

function update_pad!(
  pad::PreallocatedData_K2_5Krylov{T},
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

  pad.D .= -pad.regu.ρ
  pad.D[id.ilow] .-= pt.s_l ./ itd.x_m_lvar
  pad.D[id.iupp] .-= pt.s_u ./ itd.uvar_m_x

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
