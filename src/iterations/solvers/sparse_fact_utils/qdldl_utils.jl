using .QDLDL

get_uplo(fact_alg::QDLDLFact) = :U

mutable struct QDLDLFactorization{T} <: FactorizationData{T}
  F::QDLDL.QDLDLFactorisation{T, Int}
  initialized::Bool
  diagindK::Vector{Int}
  factorized::Bool
end

init_fact(K::Symmetric{T, SparseMatrixCSC{T, Int}}, fact_alg::QDLDLFact) where {T} =
  QDLDLFactorization(QDLDL.qdldl(K.data), false, get_diagind_K(K, :U), true)

function generic_factorize!(
  K::Symmetric{T, SparseMatrixCSC{T, Int}},
  K_fact::QDLDLFactorization{T},
) where {T}
  if K_fact.initialized
    QDLDL.update_values!(K_fact.F, K_fact.diagindK, view(K.data.nzval, K_fact.diagindK))
    K_fact.factorized = false
    try
      QDLDL.refactor!(K_fact.F)
      K_fact.factorized = true
    catch
      K_fact.factorized = false
    end
  end
  !K_fact.initialized && (K_fact.initialized = true)
end

LDLFactorizations.factorized(K_fact::QDLDLFactorization) = K_fact.factorized

function ldiv!(K_fact::QDLDLFactorization, dd::AbstractVector)
  QDLDL.solve!(K_fact.F, dd)
end
