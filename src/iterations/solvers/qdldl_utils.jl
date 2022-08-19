using .QDLDL

get_uplo(fact_alg::QDLDLFact) = :U

mutable struct QDLDLFactorization{T}
  F::QDLDL.QDLDLFactorisation{T, Int}
  initialized::Bool
  diagindK::Vector{Int}
end

init_fact(K::Symmetric{T, SparseMatrixCSC{T, Int}}, fact_alg::QDLDLFact) where {T} =
  QDLDLFactorization(QDLDL.qdldl(K.data), false, get_diagind_K(K))

function generic_factorize!(K::Symmetric{T, SparseMatrixCSC{T, Int}}, K_fact::QDLDLFactorization{T}) where {T}
  QDLDL.update_values!(K_fact.F, K_fact.diagindK, view(K.data.nzval, K_fact.diagindK))
  QDLDL.refactor!(K_fact.F)
end

LDLFactorizations.factorized(K_fact::QDLDLFactorization) = true

function ldiv!(K_fact::QDLDLFactorization, dd::AbstractVector)
  QDLDL.solve!(K_fact.F, dd)
end
