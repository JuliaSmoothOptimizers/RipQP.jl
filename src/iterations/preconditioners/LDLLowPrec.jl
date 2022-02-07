mutable struct LDLLowPrecData{T <: Real, S, Tlow, F, Ftu, Faw} <: PreconditionerDataK2{T, S}
  K::SparseMatrixCSC{Tlow, Int}
  regu::Regularization{Tlow}
  diag_Q::SparseVector{T, Int} # Q diagonal
  diagind_K::Vector{Int} # diagonal indices of J
  K_fact::LDLFactorizations.LDLFactorization{Tlow, Int, Int, Int} # factorized matrix
  fact_fail::Bool # true if factorization failed 
  P::LinearOperator{Tlow, Int, F, Ftu, Faw}
end

types_linop(op::LinearOperator{T, I, F, Ftu, Faw, S}) where {T, I, F, Ftu, Faw, S} =
  T, I, F, Ftu, Faw, S


function LDLLowPrec(
  id::QM_IntData,
  fd::QM_FloatData{T},
  regu::Regularization{T},
  D::AbstractVector{T},
  K::LinearOperator{T},
) where {T <: Real}
  @assert fd.uplo == :U
  Tlow = Float32
  diag_Q = get_diag_Q(fd.Q.data.colptr, fd.Q.data.rowval, fd.Q.data.nzval, id.nvar)
  regu_precond = Regularization(-Tlow(D[1]), Tlow(max(regu.δ, sqrt(eps(Tlow)))), sqrt(eps(Tlow)), sqrt(eps(Tlow)), :dynamic)
  K = create_K2(id, D, fd.Q.data, fd.A, diag_Q, regu_precond, T = Tlow)
  diagind_K = get_diag_sparseCSC(K.colptr, id.ncon + id.nvar)
  K_fact = ldl_analyze(Symmetric(K, :U))
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
    (res, v, α, β) -> ldiv!(res, K_fact, v),
    (res, v, α, β) -> ldiv!(res, K_fact, v),
  )
  Tlow2, I, F, Ftu, Faw, S = types_linop(P)
  return LDLLowPrecData{T, typeof(fd.c), Tlow, F, Ftu, Faw}(K, regu_precond, diag_Q, diagind_K, K_fact, false, P)
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

  # pad.pdat.regu.ρ, pad.pdat.regu.δ = sqrt(eps(Float32)), sqrt(eps(Float32))
  # pad.pdat.regu.ρ, pad.pdat.regu.δ = pad.regu.ρ, pad.regu.δ
  # update_regu!(pad.pdat.regu)

  # println(pad.pdat.regu.δ)
  out = factorize_K2!(
    pad.pdat.K,
    pad.pdat.K_fact,
    pad.D,
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
    eltype(pad.pdat.K),
    Float32,
  ) # update D and factorize K

  if out == 1
    pad.pdat.fact_fail = true
    return out
  end
  pad.pdat.K_fact.d .= abs.(pad.pdat.K_fact.d)
end