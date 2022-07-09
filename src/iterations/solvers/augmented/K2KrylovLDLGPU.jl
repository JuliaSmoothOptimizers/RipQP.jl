export LDLGPU

"""
    preconditioner = LDLGPU(; T = Float32, pos = :C, warm_start = true)

Preconditioner for [`K2KrylovParams`](@ref) using a LDL factorization in precision `T`.
The `pos` argument is used to choose the type of preconditioning with an unsymmetric Krylov method.
It can be `:C` (center), `:L` (left) or `:R` (right).
The `warm_start` argument tells RipQP to solve the system with the LDL factorization before using the Krylov method with the LDLFactorization as a preconditioner.
"""
mutable struct LDLGPU{FloatType <: DataType} <: AbstractPreconditioner
  T::FloatType
  pos::Symbol # :L (left), :R (right) or :C (center)
  warm_start::Bool
end

LDLGPU(; T::DataType = Float32, pos = :C, warm_start = true) = LDLGPU(T, pos, warm_start)

mutable struct LDLGPUData{T <: Real, S, Tlow, Op <: Union{LinearOperator, LRPrecond}} <:
               PreconditionerData{T, S}
  K::Symmetric{T, SparseMatrixCSC{T, Int}}
  L::UnitLowerTriangular{Tlow, CUDA.CUSPARSE.CuSparseMatrixCSC{Tlow, Int32}} # T or Tlow?
  d::CUDA.CuVector{Tlow}
  tmp_res::CUDA.CuVector{Tlow}
  tmp_v::CUDA.CuVector{Tlow}
  regu::Regularization{Tlow}
  K_fact::LDLFactorizations.LDLFactorization{Tlow, Int, Int, Int} # factorized matrix
  fact_fail::Bool # true if factorization failed
  warm_start::Bool
  P::Op
end

precond_name(pdat::LDLGPUData{T, S, Tlow}) where {T, S, Tlow} =
  string(Tlow, " ", string(typeof(pdat).name.name)[1:(end - 4)])

lowtype(pdat::LDLGPUData{T, S, Tlow}) where {T, S, Tlow} = Tlow

function ldl_ldiv_gpu!(res, L, d, x, P, Pinv, tmp_res, tmp_v)
  @views copyto!(tmp_res, x[P])
  ldiv!(tmp_v, L, tmp_res)
  tmp_v ./= d
  ldiv!(tmp_res, L', tmp_v)
  @views copyto!(res, tmp_res[Pinv])
end

function PreconditionerData(
  sp::AugmentedKrylovParams{<:LDLGPU},
  id::QM_IntData,
  fd::QM_FloatData{T},
  regu::Regularization{T},
  D::AbstractVector{T},
  K,
) where {T <: Real}
  Tlow = sp.preconditioner.T
  @assert fd.uplo == :U
  regu_precond = Regularization(
    -Tlow(eps(Tlow)^(3 / 4)),
    Tlow(eps(Tlow)^(0.45)),
    sqrt(eps(Tlow)),
    sqrt(eps(Tlow)),
    :dynamic,
  )

  K_fact = @timeit_debug to "LDL analyze" ldl_analyze(K, Tf = Tlow)
  K_fact.r1, K_fact.r2 = regu_precond.ρ, regu_precond.δ
  K_fact.tol = Tlow(eps(Tlow))
  K_fact.n_d = id.nvar
  ldl_factorize!(K, K_fact)
  L = UnitLowerTriangular(CUDA.CUSPARSE.CuSparseMatrixCSC(
    CuVector(K_fact.Lp),
    CuVector(K_fact.Li),
    CuVector{Tlow}(K_fact.Lx),
    size(K),
  ))
  d = abs.(CuVector(K_fact.d))
  tmp_res = similar(d)
  tmp_v = similar(d)

  P = LinearOperator(
    Tlow,
    id.nvar + id.ncon,
    id.nvar + id.ncon,
    true,
    true,
    (res, v, α, β) -> ldl_ldiv_gpu!(res, L, d, v, K_fact.P, K_fact.pinv, tmp_res, tmp_v),
  )

  return LDLGPUData{T, typeof(fd.c), Tlow, typeof(P)}(
    K,
    L,
    d,
    tmp_res,
    tmp_v,
    regu_precond,
    K_fact,
    false,
    sp.preconditioner.warm_start,
    P,
  )
end

function update_preconditioner!(
  pdat::LDLGPUData{T},
  pad::PreallocatedData{T},
  itd::IterData{T},
  pt::Point{T},
  id::QM_IntData,
  fd::Abstract_QM_FloatData{T},
  cnts::Counters,
) where {T <: Real}
  Tlow = lowtype(pad.pdat)
  pad.pdat.regu.ρ, pad.pdat.regu.δ =
    max(pad.regu.ρ, sqrt(eps(Tlow))), max(pad.regu.ρ, sqrt(eps(Tlow)))

  out = factorize_scale_K2!(
    pad.pdat.K.data,
    pad.pdat.K_fact,
    pad.D,
    pad.mt.Deq,
    pad.mt.diag_Q,
    pad.mt.diagind_K,
    pad.pdat.regu,
    pt.s_l,
    pt.s_u,
    itd.x_m_lvar,
    itd.uvar_m_x,
    id.ilow,
    id.iupp,
    id.ncon,
    id.nvar,
    cnts,
    itd.qp,
    Tlow,
    Tlow,
  ) # update D and factorize K

  copyto!(pad.pdat.L.data.nzVal, pad.pdat.K_fact.Lx)
  copyto!(pad.pdat.d, pad.pdat.K_fact.d)
  pad.pdat.d .= abs.(pad.pdat.d)

  if out == 1
    pad.pdat.fact_fail = true
    return out
  end
  if pad.pdat.warm_start
    ldl_ldiv_gpu!(pad.KS.x, pdat.L, pdat.d, pad.rhs, pdat.K_fact.P, pdat.K_fact.pinv, pdat.tmp_res, pdat.tmp_v)
    warm_start!(pad.KS, pad.KS.x)
  end
end

mutable struct K2KrylovGPUParams{T, PT} <: AugmentedKrylovParams{PT}
  uplo::Symbol
  kmethod::Symbol
  preconditioner::PT
  rhs_scale::Bool
  equilibrate::Bool
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
  verbose::Int
end

function K2KrylovGPUParams{T}(;
  uplo::Symbol = :U,
  kmethod::Symbol = :minres,
  preconditioner::AbstractPreconditioner = Identity(),
  rhs_scale::Bool = true,
  equilibrate::Bool = true,
  atol0::T = T(1.0e-4),
  rtol0::T = T(1.0e-4),
  atol_min::T = T(1.0e-10),
  rtol_min::T = T(1.0e-10),
  ρ0::T = T(sqrt(eps()) * 1e5),
  δ0::T = T(sqrt(eps()) * 1e5),
  ρ_min::T = 1e2 * sqrt(eps(T)),
  δ_min::T = 1e2 * sqrt(eps(T)),
  itmax::Int = 0,
  mem::Int = 20,
  verbose::Int = 0,
) where {T <: Real}
  return K2KrylovGPUParams(
    uplo,
    kmethod,
    preconditioner,
    rhs_scale,
    equilibrate,
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
    verbose,
  )
end

K2KrylovGPUParams(; kwargs...) = K2KrylovGPUParams{Float64}(; kwargs...)

mutable struct MatrixTools{T}
  diag_Q::SparseVector{T, Int} # Q diag
  diagind_K::Vector{Int}
  Deq::Diagonal{T, Vector{T}}
  C_eq::Diagonal{T, Vector{T}}
end

convert(::Type{MatrixTools{T}}, mt::MatrixTools) where {T} = MatrixTools(
  convert(SparseVector{T, Int}, mt.diag_Q),
  mt.diagind_K,
  Diagonal(convert(Vector{T}, mt.Deq.diag)),
  Diagonal(convert(Vector{T}, mt.C_eq.diag)),
)

mutable struct PreallocatedDataK2KrylovGPU{
  T <: Real,
  S,
  M <: Union{LinearOperator{T}, AbstractMatrix{T}},
  MT <: Union{MatrixTools{T}, Int},
  Pr <: PreconditionerData,
  Ksol <: KrylovSolver,
} <: PreallocatedDataAugmentedKrylov{T, S}
  pdat::Pr
  D::S                                  # temporary top-left diagonal
  Dcpu::Vector{T}
  rhs::S
  rhs_scale::Bool
  equilibrate::Bool
  regu::Regularization{T}
  δv::Vector{T}
  K::M # augmented matrix
  Kcpu::Symmetric{T, SparseMatrixCSC{T, Int}}
  diagQnz::S
  mt::MT
  deq::S
  KS::Ksol
  atol::T
  rtol::T
  atol_min::T
  rtol_min::T
  itmax::Int
  verbose::Int
end

function opK2eqprod!(
  res::AbstractVector{T},
  nvar::Int,
  Q::Union{AbstractMatrix{T}, AbstractLinearOperator{T}},
  D::AbstractVector{T},
  A::Union{AbstractMatrix{T}, AbstractLinearOperator{T}},
  δv::AbstractVector{T},
  deq::AbstractVector{T},
  v::AbstractVector{T},
  vtmp::AbstractVector{T},
  uplo::Symbol,
) where {T}
  vtmp .= deq .* v
  @views mul!(res[1:nvar], Q, vtmp[1:nvar], -one(T), zero(T))
  res[1:nvar] .+= D .* vtmp[1:nvar]
  if uplo == :U
    @views mul!(res[1:nvar], A, vtmp[(nvar + 1):end], one(T), one(T))
    @views mul!(res[(nvar + 1):end], A', vtmp[1:nvar], one(T), zero(T))
  else
    @views mul!(res[1:nvar], A', vtmp[(nvar + 1):end], one(T), one(T))
    @views mul!(res[(nvar + 1):end], A, vtmp[1:nvar], one(T), zero(T))
  end
  res[(nvar + 1):end] .+= @views δv[1] .* vtmp[(nvar + 1):end]
  res .*= deq
end

function PreallocatedData(
  sp::K2KrylovGPUParams,
  fd::QM_FloatData{T},
  id::QM_IntData,
  itd::IterData{T},
  pt::Point{T},
  iconf::InputConfig{Tconf},
) where {T <: Real, Tconf <: Real}

  @assert fd.uplo == :U
  # init Regularization values
  D = similar(fd.c, id.nvar)
  if iconf.mode == :mono
    regu = Regularization(T(sp.ρ0), T(sp.δ0), T(sp.ρ_min), T(sp.δ_min), :classic)
    D .= -T(1.0e0) / 2
  else
    regu = Regularization(T(sp.ρ0), T(sp.δ0), T(sp.ρ_min), T(sp.δ_min), :hybrid)
    D .= -T(1.0e-2)
  end
  δv = [regu.δ] # put it in a Vector so that we can modify it without modifying opK2prod!
  Qcpu = SparseMatrixCSC(fd.Q.data)
  Acpu = SparseMatrixCSC(fd.A)
  diag_Q = get_diag_Q(Qcpu.colptr, Qcpu.rowval, Qcpu.nzval, id.nvar)
  Kcpu = Symmetric(create_K2(id, Vector(D), Qcpu, Acpu, diag_Q, regu), fd.uplo)
  diagind_K = get_diag_sparseCSC(Kcpu.data.colptr, id.ncon + id.nvar)
  if sp.equilibrate
    Deq = Diagonal(Vector{T}(undef, id.nvar + id.ncon))
    Deq.diag .= one(T)
    C_eq = Diagonal(Vector{T}(undef, id.nvar + id.ncon))
  else
    Deq = Diagonal(Vector{T}(undef, 0))
    C_eq = Diagonal(Vector{T}(undef, 0))
  end
  mt = MatrixTools(diag_Q, diagind_K, Deq, C_eq)
  Dcpu = Vector(D)
  deq = CuVector(Deq.diag)
  vtmp = similar(deq)
  K = LinearOperator(
    T,
    id.nvar + id.ncon,
    id.nvar + id.ncon,
    true,
    true,
    (res, v, α, β) -> opK2eqprod!(res, id.nvar, fd.Q, D, fd.A, δv, deq, v, vtmp, fd.uplo),
  )
  diagQnz = similar(deq, length(diag_Q.nzval))
  copyto!(diagQnz, diag_Q.nzval)

  rhs = similar(fd.c, id.nvar + id.ncon)
  KS = @timeit_debug to "krylov solver setup" init_Ksolver(K, rhs, sp)
  pdat = @timeit_debug to "preconditioner setup" PreconditionerData(sp, id, fd, regu, D, Kcpu)

  return PreallocatedDataK2KrylovGPU(
    pdat,
    D,
    Dcpu,
    rhs,
    sp.rhs_scale,
    sp.equilibrate,
    regu,
    δv,
    K, #K
    Kcpu,
    diagQnz,
    mt,
    deq,
    KS,
    T(sp.atol0),
    T(sp.rtol0),
    T(sp.atol_min),
    T(sp.rtol_min),
    sp.itmax,
    sp.verbose,
  )
end

function solver!(
  dd::AbstractVector{T},
  pad::PreallocatedDataK2KrylovGPU{T},
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
  pad.rhs .= pad.equilibrate ? dd .* pad.deq : dd
  if pad.rhs_scale
    rhsNorm = kscale!(pad.rhs)
  end
  if step !== :cc
    @timeit_debug to "preconditioner update" update_preconditioner!(
      pad.pdat,
      pad,
      itd,
      pt,
      id,
      fd,
      cnts,
    )
  end
  @timeit_debug to "Krylov solve" ksolve!(
    pad.KS,
    pad.K,
    pad.rhs,
    pad.pdat.P,
    verbose = pad.verbose,
    atol = pad.atol,
    rtol = pad.rtol,
    itmax = pad.itmax,
  )
  update_kresiduals_history!(res, pad.K, pad.KS.x, pad.rhs)
  if pad.rhs_scale
    kunscale!(pad.KS.x, rhsNorm)
  end
  if pad.equilibrate
    if typeof(pad.Kcpu) <: Symmetric{T, SparseMatrixCSC{T, Int}} && step !== :aff
      rdiv!(pad.Kcpu.data, pad.mt.Deq)
      ldiv!(pad.mt.Deq, pad.Kcpu.data)
    end
    pad.KS.x .*= pad.deq
  end
  dd .= pad.KS.x

  return 0
end

function update_pad!(
  pad::PreallocatedDataK2KrylovGPU{T},
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
  pad.D[pad.mt.diag_Q.nzind] .-= pad.diagQnz
  copyto!(pad.Dcpu, pad.D)
  pad.Kcpu.data.nzval[view(pad.mt.diagind_K, 1:(id.nvar))] = pad.Dcpu
  pad.Kcpu.data.nzval[view(pad.mt.diagind_K, (id.nvar + 1):(id.ncon + id.nvar))] .= pad.regu.δ
  if pad.equilibrate
    pad.mt.Deq.diag .= one(T)
    @timeit_debug to "equilibration" equilibrate!(
      pad.Kcpu,
      pad.mt.Deq,
      pad.mt.C_eq;
      ϵ = T(1.0e-2),
      max_iter = 15,
    )
    copyto!(pad.deq, pad.mt.Deq.diag)
  end

  return 0
end

# function get_mat_QPData(A::CUDA.CUSPARSE.CuSparseMatrixCSC, H, nvar::Int, ncon::Int, uplo::Symbol)
#   fdA = uplo == :U ? CUDA.CUSPARSE.CuSparseMatrixCSC(SparseMatrixCSC(transpose(SparseMatrixCSC(A)))) : A
#   fdH = uplo == :U ? CUDA.CUSPARSE.CuSparseMatrixCSC(SparseMatrixCSC(transpose(SparseMatrixCSC(H)))) : H
#   return fdA, Symmetric(fdH, uplo)
# end

function get_mat_QPData(
  A::CUDA.CUSPARSE.CuSparseMatrixCOO{T, Ti},
  H::CUDA.CUSPARSE.CuSparseMatrixCOO{T, Ti},
  nvar::Int,
  ncon::Int,
  uplo::Symbol,
) where {T, Ti}
  # A is Aᵀ of QuadraticModel QM
  fdA = CUDA.CUSPARSE.CuSparseMatrixCSC(sparse(Vector(A.colInd), Vector(A.rowInd), Vector(A.nzVal), nvar, ncon))
  fdQ = CUDA.CUSPARSE.CuSparseMatrixCSC(sparse(Vector(H.colInd), Vector(H.rowInd), Vector(H.nzVal), nvar, nvar))
  return fdA, Symmetric(fdQ, uplo)
end