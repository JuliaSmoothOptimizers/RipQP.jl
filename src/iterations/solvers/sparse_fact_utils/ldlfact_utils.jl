get_uplo(fact_alg::LDLFact) = :U

init_fact(K::Symmetric{T, SparseMatrixCSC{T, Int}}, fact_alg::LDLFact) where {T} = ldl_analyze(K)

generic_factorize!(K::Symmetric, K_fact::LDLFactorizations.LDLFactorization) =
  ldl_factorize!(K, K_fact)

  # LDLFactorization conversion function
convertldl(T::DataType, K_fact::LDLFactorizations.LDLFactorization) = LDLFactorizations.LDLFactorization(
  K_fact.__analyzed,
  K_fact.__factorized,
  K_fact.__upper,
  K_fact.n,
  K_fact.parent,
  K_fact.Lnz,
  K_fact.flag,
  K_fact.P,
  K_fact.pinv,
  K_fact.Lp,
  K_fact.Cp,
  K_fact.Ci,
  K_fact.Li,
  convert(Array{T}, K_fact.Lx),
  convert(Array{T}, K_fact.d),
  convert(Array{T}, K_fact.Y),
  K_fact.pattern,
  T(K_fact.r1),
  T(K_fact.r2),
  T(K_fact.tol),
  K_fact.n_d,
)