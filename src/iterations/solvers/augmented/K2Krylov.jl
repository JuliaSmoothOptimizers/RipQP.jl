export K2KrylovParams

"""
Type to use the K2 formulation with a Krylov method, using the package 
[`Krylov.jl`](https://github.com/JuliaSmoothOptimizers/Krylov.jl). 
The outer constructor 

    K2KrylovParams(; uplo = :L, kmethod = :minres, preconditioner = :Identity,
                   rhs_scale = true, form_mat = false,
                   atol0 = 1.0e-4, rtol0 = 1.0e-4, 
                   atol_min = 1.0e-10, rtol_min = 1.0e-10,
                   ρ0 = sqrt(eps()) * 1e5, δ0 = sqrt(eps()) * 1e5, 
                   ρ_min = 1e2 * sqrt(eps()), δ_min = 1e2 * sqrt(eps()),
                   memory = 20)

creates a [`RipQP.SolverParams`](@ref) that should be used to create a [`RipQP.InputConfig`](@ref).
The available methods are:
- `:minres`
- `:minres_qlp`
- `:symmlq`

The list of available preconditioners for this solver is displayed here: [`RipQP.PreconditionerData`](@ref).
"""
mutable struct K2KrylovParams <: AugmentedParams
  uplo::Symbol
  kmethod::Symbol
  preconditioner::Symbol
  rhs_scale::Bool
  form_mat::Bool
  atol0::Float64
  rtol0::Float64
  atol_min::Float64
  rtol_min::Float64
  ρ0::Float64
  δ0::Float64
  ρ_min::Float64
  δ_min::Float64
  mem::Int
end

function K2KrylovParams(;
  uplo::Symbol = :L,
  kmethod::Symbol = :minres,
  preconditioner::Symbol = :Identity,
  rhs_scale::Bool = true,
  form_mat::Bool = false,
  atol0::T = 1.0e-4,
  rtol0::T = 1.0e-4,
  atol_min::T = 1.0e-10,
  rtol_min::T = 1.0e-10,
  ρ0::T = sqrt(eps()) * 1e5,
  δ0::T = sqrt(eps()) * 1e5,
  ρ_min::T = 1e2 * sqrt(eps()),
  δ_min::T = 1e2 * sqrt(eps()),
  mem::Int = 20,
) where {T <: Real}
  if uplo == :L && form_mat
    error("matrix can only be created if uplo == :U")
  end
  return K2KrylovParams(
    uplo,
    kmethod,
    preconditioner,
    rhs_scale,
    form_mat,
    atol0,
    rtol0,
    atol_min,
    rtol_min,
    ρ0,
    δ0,
    ρ_min,
    δ_min,
    mem,
  )
end

mutable struct MatrixTools{T}
  diag_Q::SparseVector{T, Int} # Q diag
  diagind_K::Vector{Int}
end

mutable struct PreallocatedDataK2Krylov{
  T <: Real,
  S,
  M <: Union{LinearOperator{T}, AbstractMatrix{T}},
  MT <: Union{MatrixTools{T}, Int},
  Pr <: PreconditionerData,
  Ksol <: KrylovSolver,
} <: PreallocatedDataAugmentedKrylov{T, S}
  pdat::Pr
  D::S                                  # temporary top-left diagonal
  rhs::S
  rhs_scale::Bool
  regu::Regularization{T}
  δv::Vector{T}
  K::M # augmented matrix
  mt::MT
  KS::Ksol
  atol::T
  rtol::T
  atol_min::T
  rtol_min::T
end

function opK2prod!(
  res::AbstractVector{T},
  nvar::Int,
  Q::Union{AbstractMatrix{T}, AbstractLinearOperator{T}},
  D::AbstractVector{T},
  A::Union{AbstractMatrix{T}, AbstractLinearOperator{T}},
  δv::AbstractVector{T},
  v::AbstractVector{T},
  α::T,
  β::T,
  uplo::Symbol,
) where {T}
  @views mul!(res[1:nvar], Q, v[1:nvar], -α, β)
  res[1:nvar] .+= α .* D .* v[1:nvar]
  if uplo == :U
    @views mul!(res[1:nvar], A, v[(nvar + 1):end], α, one(T))
    @views mul!(res[(nvar + 1):end], A', v[1:nvar], α, β)
  else
    @views mul!(res[1:nvar], A', v[(nvar + 1):end], α, one(T))
    @views mul!(res[(nvar + 1):end], A, v[1:nvar], α, β)
  end
  res[(nvar + 1):end] .+= @views (α * δv[1]) .* v[(nvar + 1):end]
end

function PreallocatedData(
  sp::K2KrylovParams,
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
  δv = [regu.δ] # put it in a Vector so that we can modify it without modifying opK2prod!
  if sp.form_mat
    diag_Q = get_diag_Q(fd.Q.data.colptr, fd.Q.data.rowval, fd.Q.data.nzval, id.nvar)
    K = Symmetric(create_K2(id, D, fd.Q.data, fd.A, diag_Q, regu), fd.uplo)
    diagind_K = get_diag_sparseCSC(K.data.colptr, id.ncon + id.nvar)
    mt = MatrixTools(diag_Q, diagind_K)
  else
    K = LinearOperator(
      T,
      id.nvar + id.ncon,
      id.nvar + id.ncon,
      true,
      true,
      (res, v, α, β) -> opK2prod!(res, id.nvar, fd.Q, D, fd.A, δv, v, α, β, fd.uplo),
    )
    mt = 0
  end

  rhs = similar(fd.c, id.nvar + id.ncon)
  KS = init_Ksolver(K, rhs, sp)
  pdat = eval(sp.preconditioner)(id, fd, regu, D, K)

  return PreallocatedDataK2Krylov(
    pdat,
    D,
    rhs,
    sp.rhs_scale,
    regu,
    δv,
    K, #K
    mt,
    KS,
    T(sp.atol0),
    T(sp.rtol0),
    T(sp.atol_min),
    T(sp.rtol_min),
  )
end

function solver!(
  dd::AbstractVector{T},
  pad::PreallocatedDataK2Krylov{T},
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
  pad.rhs .= dd
  if pad.rhs_scale
    rhsNorm = kscale!(pad.rhs)
  end
  ksolve!(pad.KS, pad.K, pad.rhs, pad.pdat.P, verbose = 0, atol = pad.atol, rtol = pad.rtol)
  update_kresiduals_history!(res, pad.K, pad.KS.x, pad.rhs)
  if pad.rhs_scale
    kunscale!(pad.KS.x, rhsNorm)
  end

  dd .= pad.KS.x

  return 0
end

function update_pad!(
  pad::PreallocatedDataK2Krylov{T},
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

  pad.D .= -pad.regu.ρ
  pad.D[id.ilow] .-= pt.s_l ./ itd.x_m_lvar
  pad.D[id.iupp] .-= pt.s_u ./ itd.uvar_m_x
  pad.δv[1] = pad.regu.δ
  if typeof(pad.K) <: Symmetric{T, SparseMatrixCSC{T, Int}}
    pad.D[pad.mt.diag_Q.nzind] .-= pad.mt.diag_Q.nzval
    pad.K.data.nzval[view(pad.mt.diagind_K, 1:(id.nvar))] = pad.D
    pad.K.data.nzval[view(pad.mt.diagind_K, (id.nvar + 1):(id.ncon + id.nvar))] .= pad.regu.δ
  end

  update_preconditioner!(pad.pdat, pad, itd, pt, id, fd, cnts)

  return 0
end
