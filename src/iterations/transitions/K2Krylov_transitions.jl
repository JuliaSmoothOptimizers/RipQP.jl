function convertpad(
  ::Type{<:PreallocatedData{T}},
  pad::PreallocatedDataK2LDL{T_old},
  sp_old::K2LDLParams,
  sp_new::K2KrylovParams,
  id::QM_IntData,
  fd::Abstract_QM_FloatData,
) where {T <: Real, T_old <: Real}
  D = convert(Array{T}, pad.D)
  regu = convert(Regularization{T}, pad.regu)
  regu.ρ_min = T(sp_new.ρ_min)
  regu.δ_min = T(sp_new.δ_min)
  K = Symmetric(convert(eval(typeof(pad.K.data).name.name){T, Int}, pad.K.data), sp_new.uplo)
  rhs = similar(D, id.nvar + id.ncon)
  δv = [regu.δ]
  if sp_new.equilibrate
    Deq = Diagonal(similar(D, id.nvar + id.ncon))
    Deq.diag .= one(T)
    C_eq = Diagonal(similar(D, id.nvar + id.ncon))
  else
    Deq = Diagonal(similar(D, 0))
    C_eq = Diagonal(similar(D, 0))
  end
  mt = MatrixTools(convert(SparseVector{T, Int}, pad.diag_Q), pad.diagind_K, Deq, C_eq)
  regu_precond = pad.regu
  regu_precond.regul = :dynamic
  pdat = PreconditionerData(sp_new, pad.K_fact, id.nvar, id.ncon, regu_precond, K)
  KS = init_Ksolver(K, rhs, sp_new)

  return PreallocatedDataK2Krylov(
    pdat,
    D,
    rhs,
    sp_new.rhs_scale,
    sp_new.equilibrate,
    regu,
    δv,
    K, #K
    mt,
    KS,
    0,
    T(sp_new.atol0),
    T(sp_new.rtol0),
    T(sp_new.atol_min),
    T(sp_new.rtol_min),
    sp_new.itmax,
  )
end

function convertpad(
  ::Type{<:PreallocatedData{T}},
  pad::PreallocatedDataK2Krylov{T_old},
  sp_old::K2KrylovParams,
  sp_new::K2KrylovParams,
  id::QM_IntData,
  fd::Abstract_QM_FloatData,
) where {T <: Real, T_old <: Real}
  D = convert(Array{T}, pad.D)
  regu = convert(Regularization{T}, pad.regu)
  regu.ρ_min = T(sp_new.ρ_min)
  regu.δ_min = T(sp_new.δ_min)
  K = Symmetric(convert(eval(typeof(pad.K.data).name.name){T, Int}, pad.K.data), sp_new.uplo)
  rhs = similar(D, id.nvar + id.ncon)
  δv = [regu.δ]
  if !(sp_old.equilibrate) && sp_new.equilibrate
    Deq = Diagonal(similar(D, id.nvar + id.ncon))
    C_eq = Diagonal(similar(D, id.nvar + id.ncon))
    mt = MatrixTools(convert(SparseVector{T, Int}, pad.mt.diag_Q), pad.mt.diagind_K, Deq, C_eq)
  else
    mt = convert(MatrixTools{T}, pad.mt)
    mt.Deq.diag .= one(T)
  end
  sp_new.equilibrate && (mt.Deq.diag .= one(T))
  regu_precond = convert(Regularization{sp_new.preconditioner.T}, pad.regu)
  regu_precond.regul = :dynamic
  if typeof(sp_old.preconditioner.fact_alg) == LLDLFact ||
     typeof(sp_old.preconditioner.fact_alg) != typeof(sp_new.preconditioner.fact_alg)
    if get_uplo(sp_old.preconditioner.fact_alg) != get_uplo(sp_new.preconditioner.fact_alg)
      # transpose if new factorization does not use the same uplo
      K = Symmetric(SparseMatrixCSC(K.data'), sp_new.uplo)
      mt.diagind_K = get_diagind_K(K, sp_new.uplo)
    end
    K_fact = init_fact(K, sp_new.preconditioner.fact_alg)
  else
    K_fact = convertldl(sp_new.preconditioner.T, pad.pdat.K_fact)
  end
  pdat = PreconditionerData(sp_new, K_fact, id.nvar, id.ncon, regu_precond, K)
  KS = init_Ksolver(K, rhs, sp_new)

  return PreallocatedDataK2Krylov(
    pdat,
    D,
    rhs,
    sp_new.rhs_scale,
    sp_new.equilibrate,
    regu,
    δv,
    K, #K
    mt,
    KS,
    0,
    T(sp_new.atol0),
    T(sp_new.rtol0),
    T(sp_new.atol_min),
    T(sp_new.rtol_min),
    sp_new.itmax,
  )
end

function convertpad(
  ::Type{<:PreallocatedData{T}},
  pad::PreallocatedDataK2Krylov{T_old},
  sp_old::K2KrylovParams,
  sp_new::K2LDLParams,
  id::QM_IntData,
  fd::Abstract_QM_FloatData,
) where {T <: Real, T_old <: Real}
  D = convert(Array{T}, pad.D)
  regu = convert(Regularization{T}, pad.regu)
  regu.ρ_min = T(sp_new.ρ_min)
  regu.δ_min = T(sp_new.δ_min)
  regu.ρ *= 10
  regu.δ *= 10
  regu.regul = sp_new.fact_alg.regul
  K_fact = convertldl(T, pad.pdat.K_fact)
  K = Symmetric(convert(eval(typeof(pad.K.data).name.name){T, Int}, pad.K.data), sp_new.uplo)

  return PreallocatedDataK2LDL(
    D,
    regu,
    sp_new.safety_dist_bnd,
    convert(SparseVector{T, Int}, pad.mt.diag_Q), #diag_Q
    K, #K
    K_fact, #K_fact
    false,
    pad.mt.diagind_K, #diagind_K
  )
end
