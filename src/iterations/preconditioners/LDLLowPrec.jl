mutable struct LDLLowPrecData{T <: Real, S, Tlow, F, Ftu, Faw} <: PreconditionerData{T, S}
  K::SparseMatrixCSC{Tlow, Int}
  Dlp::S
  regu::Regularization{Tlow}
  diag_Q::SparseVector{T, Int} # Q diagonal
  diagind_K::Vector{Int} # diagonal indices of J
  K_fact::LDLFactorizations.LDLFactorization{Tlow, Int, Int, Int} # factorized matrix
  fact_fail::Bool # true if factorization failed 
  P::LinearOperator{Tlow, Int, F, Ftu, Faw}
end

types_linop(op::LinearOperator{T, I, F, Ftu, Faw, S}) where {T, I, F, Ftu, Faw, S} =
  T, I, F, Ftu, Faw, S

lowtype(pdat::LDLLowPrecData{T, S, Tlow}) where {T, S, Tlow} = Tlow

LDLLowPrec32(id::QM_IntData, fd::QM_FloatData{T}, regu::Regularization{T}, D::AbstractVector{T}, K) where {T} = 
  LDLLowPrec(id, fd, regu, D, K, Float32)

function LDLLowPrec(
  id::QM_IntData,
  fd::QM_FloatData{T},
  regu::Regularization{T},
  D::AbstractVector{T},
  K,
  Tlow::DataType,
) where {T <: Real}
  @assert fd.uplo == :U
  diag_Q = get_diag_Q(fd.Q.data.colptr, fd.Q.data.rowval, fd.Q.data.nzval, id.nvar)
  regu_precond = Regularization(-Tlow(D[1]), Tlow(max(regu.δ, sqrt(eps(Tlow)))), sqrt(eps(Tlow)), sqrt(eps(Tlow)), :classic)
  K = create_K2(id, D, fd.Q.data, fd.A, diag_Q, regu_precond, T = Tlow)
  Dlp = copy(D)
  diagind_K = get_diag_sparseCSC(K.colptr, id.ncon + id.nvar)
  K_fact = ldl_analyze(Symmetric(K, :U))
  regu_precond.regul = :dynamic
  if regu_precond.regul == :dynamic
    Amax = @views norm(K.nzval[diagind_K], Inf)
    regu_precond.ρ, regu_precond.δ = -Tlow(eps(Tlow)^(3 / 4)), Tlow(eps(Tlow)^(0.45))
    K_fact.r1, K_fact.r2 = regu_precond.ρ, regu_precond.δ
    K_fact.tol = Amax * Tlow(eps(Tlow))
    K_fact.n_d = id.nvar
  end
  ldl_factorize!(Symmetric(K, :U), K_fact)
  K_fact.d .= abs.(K_fact.d)
  P = LinearOperator(Tlow,
    id.nvar + id.ncon,
    id.nvar + id.ncon,
    true,
    true,
    (res, v, α, β) -> ldiv!(res, K_fact, v),
  )
  Tlow2, I, F, Ftu, Faw, S = types_linop(P)
  return LDLLowPrecData{T, typeof(fd.c), Tlow, F, Ftu, Faw}(K, Dlp, regu_precond, diag_Q, diagind_K, K_fact, false, P)
end

function factorize_scale_K2!(
  K,
  K_fact,
  D,
  Deq,
  diag_Q,
  diagind_K,
  regu,
  s_l,
  s_u,
  x_m_lvar,
  uvar_m_x,
  ilow,
  iupp,
  ncon,
  nvar,
  cnts,
  qp,
  T,
  T0,
)

  if regu.regul == :dynamic
    update_K_dynamic!(K, K_fact, regu, diagind_K, cnts, T, qp)
    ldl_factorize!(Symmetric(K, :U), K_fact)
  elseif regu.regul == :classic
    ldl_factorize!(Symmetric(K, :U), K_fact)
    while !factorized(K_fact)
      out = update_regu_trycatch!(regu, cnts, T, T0)
      out == 1 && return out
      cnts.c_catch += 1
      cnts.c_catch >= 4 && return 1
      update_K!(K, D, regu, s_l, s_u, x_m_lvar, uvar_m_x, ilow, iupp, diag_Q, diagind_K, nvar, ncon, T)
      ldl_factorize!(Symmetric(K, :U), K_fact)
    end

  else # no Regularization
    ldl_factorize!(Symmetric(K, :U), K_fact)
  end

  return 0 # factorization succeeded
end


function update_preconditioner!(
  pdat::LDLLowPrecData{T},
  pad::PreallocatedData{T},
  itd::IterData{T},
  pt::Point{T},
  id::QM_IntData,
  fd::QM_FloatData{T},
  cnts::Counters,
) where {T <: Real}

  Tlow = lowtype(pad.pdat)
  pad.pdat.regu.ρ, pad.pdat.regu.δ = max(pad.regu.ρ, sqrt(eps(Tlow))), max(pad.regu.ρ, sqrt(eps(Tlow)))
  pad.pdat.K.nzval .= pad.K.data.nzval

  out = factorize_scale_K2!(
    pad.pdat.K,
    pad.pdat.K_fact,
    pad.pdat.Dlp,
    pad.mt.Deq,
    pad.pdat.diag_Q,
    pad.pdat.diagind_K,
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

  if out == 1
    pad.pdat.fact_fail = true
    return out
  end
  pad.pdat.K_fact.d .= abs.(pad.pdat.K_fact.d)
end