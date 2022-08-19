using SuiteSparse

get_uplo(fact_alg::CholmodFact) = :U

mutable struct CholmodFactorization{T}
  F::SuiteSparse.CHOLMOD.Factor{T}
  initialized::Bool
end

init_fact(K::Symmetric{T, SparseMatrixCSC{T, Int}}, fact_alg::CholmodFact) where {T} =
  CholmodFactorization(ldlt(K), false)

function generic_factorize!(K::Symmetric{T, SparseMatrixCSC{T, Int}}, K_fact::CholmodFactorization{T}) where {T}
  K_fact.initialized && ldlt!(K_fact.F, K)
  !K_fact.initialized && (K_fact.initialized = true)
end

LDLFactorizations.factorized(K_fact::CholmodFactorization) = true

function ldiv!(K_fact::CholmodFactorization, dd::AbstractVector)
  dd .= K_fact.F \ dd
end
