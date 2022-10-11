get_uplo(fact_alg::LLDLFact) = :L

mutable struct LLDLFactorizationData{T, F <: LimitedLDLFactorization} <: FactorizationData{T}
  LLDL::F
  mem::Int
  droptol::T
end

init_fact(K::Symmetric{T, SparseMatrixCSC{T, Int}}, fact_alg::LLDLFact) where {T} =
  LLDLFactorizationData(lldl(K.data, memory = fact_alg.mem), fact_alg.mem, T(fact_alg.droptol))

init_fact(K::Symmetric{T, SparseMatrixCSC{T, Int}}, fact_alg::LLDLFact, ::Type{Tf}) where {T, Tf} =
  LLDLFactorizationData(lldl(K.data, Tf, memory = fact_alg.mem), fact_alg.mem, Tf(fact_alg.droptol))

generic_factorize!(K::Symmetric, K_fact::LLDLFactorizationData) = lldl_factorize!(K_fact.LLDL, K.data)

RipQP.factorized(K_fact::LLDLFactorizationData) = LimitedLDLFactorizations.factorized(K_fact.LLDL)

ldiv!(K_fact::LLDLFactorizationData, dd::AbstractVector) = ldiv!(K_fact.LLDL, dd)

# used only for preconditioner
function abs_diagonal!(K_fact::LLDLFactorizationData)
  K_fact.LLDL.D .= abs.(K_fact.LLDL.D)
end