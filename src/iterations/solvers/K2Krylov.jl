export K2KrylovParams

"""
Type to use the K2 formulation with a Krylov method, using the package 
[`Krylov.jl`](https://github.com/JuliaSmoothOptimizers/Krylov.jl). 
The outer constructor 

    K2KrylovParams(; kmethod = :minres, preconditioner = :Jacobi, 
                   ratol = 1.0e-10, rrtol = 1.0e-10)

creates a [`RipQP.SolverParams`](@ref) that should be used to create a [`RipQP.InputConfig`](@ref).
The available methods are:
- `:minres`
- `:minres_qlp`

The list of available preconditioners for this solver is displayed here: [`RipQP.PreconditionerDataK2`](@ref).
"""
struct K2KrylovParams <: SolverParams
  kmethod::Symbol
  preconditioner::Symbol
  atol::Float64
  rtol::Float64
  ρ_min::Float64
  δ_min::Float64
end

function K2KrylovParams(;
  kmethod::Symbol = :minres,
  preconditioner::Symbol = :Jacobi,
  atol::T = 1.0e-10,
  rtol::T = 1.0e-10,
  ρ_min::T = 1e3 * sqrt(eps()),
  δ_min::T = 1e4 * sqrt(eps()),
) where {T <: Real}
  return K2KrylovParams(kmethod, preconditioner, atol, rtol, ρ_min, δ_min)
end

function opK2prod!(
  res::AbstractVector{T},
  nvar::Int,
  Q::AbstractMatrix{T},
  D::AbstractVector{T},
  AT::AbstractMatrix{T},
  δv::AbstractVector{T},
  v::AbstractVector{T},
  α::T,
  β::T,
) where {T}
  @views mul!(res[1:nvar], Q, v[1:nvar], -α, β)
  res[1:nvar] .+= α .* D .* v[1:nvar]
  @views mul!(res[1:nvar], AT, v[(nvar + 1):end], α, one(T))
  @views mul!(res[(nvar + 1):end], AT', v[1:nvar], α, β)
  res[(nvar + 1):end] .+= @views (α * δv[1]) .* v[(nvar + 1):end]
end

function PreallocatedData(
  sp::K2KrylovParams,
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
  δv = [regu.δ] # put it in a Vector so that we can modify it without modifying opK2prod!
  K = LinearOperator(
    T,
    id.nvar + id.ncon,
    id.nvar + id.ncon,
    true,
    true,
    (res, v, α, β) -> opK2prod!(res, id.nvar, fd.Q, D, fd.AT, δv, v, α, β),
  )

  rhs = similar(fd.c, id.nvar + id.ncon)
  kstring = string(sp.kmethod)
  KS = eval(KSolver(sp.kmethod))(K, rhs)

  pdat = eval(sp.preconditioner)(id, fd, regu, D, K)

  return eval(Symbol(:PreallocatedData_K2, sp.kmethod))(
    pdat,
    D,
    rhs,
    regu,
    δv,
    K, #K
    KS,
    sp.atol,
    sp.rtol,
  )
end

function solver!(
  pad::PreallocatedData_K2Krylov{T},
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
  pad.rhs .= (step == :aff) ? dda.Δxy_aff : pad.rhs .= itd.Δxy
  rhsNorm = norm(pad.rhs)
  if rhsNorm != zero(T)
    pad.rhs ./= rhsNorm
  end
  ksolve!(pad.KS, pad.K, pad.rhs, pad.pdat.P, verbose = 0, atol = pad.atol, rtol = pad.rtol)
  if typeof(res) <: ResidualsHistory
    mul!(res.KΔxy, pad.K, pad.KS.x) # krylov residuals
    res.kresNorm = norm(res.KΔxy .- pad.rhs)
  end 
  if rhsNorm != zero(T)
    pad.KS.x .*= rhsNorm
  end

  if step == :aff
    dda.Δxy_aff .= pad.KS.x
  else
    itd.Δxy .= pad.KS.x
  end

  return 0
end

function update_pad!(
  pad::PreallocatedData_K2Krylov{T},
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
  pad.δv[1] = pad.regu.δ

  update_preconditioner!(pad.pdat, pad, itd, pt, id, fd, cnts)

  return 0
end
