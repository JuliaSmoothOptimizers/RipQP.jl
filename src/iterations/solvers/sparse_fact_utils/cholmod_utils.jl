using .SuiteSparse

get_uplo(fact_alg::CholmodFact) = :U

mutable struct CholmodFactorization{T} <: FactorizationData{T}
  F::SuiteSparse.CHOLMOD.Factor{T}
  initialized::Bool
  factorized::Bool
end

init_fact(K::Symmetric{T, SparseMatrixCSC{T, Int}}, fact_alg::CholmodFact) where {T} =
  CholmodFactorization(ldlt(K), false, true)

function generic_factorize!(
  K::Symmetric{T, SparseMatrixCSC{T, Int}},
  K_fact::CholmodFactorization{T},
) where {T}
  if K_fact.initialized
    K_fact.factorized = false
    try
      ldlt!(K_fact.F, K)
      K_fact.factorized = true
    catch
      K_fact.factorized = false
    end
  end
  !K_fact.initialized && (K_fact.initialized = true)
end

LDLFactorizations.factorized(K_fact::CholmodFactorization) = K_fact.factorized

function ldiv!(K_fact::CholmodFactorization, dd::AbstractVector)
  dd .= K_fact.F \ dd
end
