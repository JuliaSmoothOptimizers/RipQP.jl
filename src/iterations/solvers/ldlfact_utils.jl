get_uplo(fact_alg::LDLFact) = :U

init_fact(K::Symmetric{T, SparseMatrixCSC{T, Int}}, fact_alg::LDLFact) where {T} = ldl_analyze(K)

generic_factorize!(K::Symmetric, K_fact::LDLFactorizations.LDLFactorization) = ldl_factorize!(K, K_fact)
