using .CUDSS

get_uplo(fact_alg::CUDSSFact) = fact_alg.view

mutable struct CUDSSFactorization{T} <: FactorizationData{T}
  F::CudssSolver{T}
  type::Symbol
  initialized::Bool
  factorized::Bool
end

init_fact(K::Symmetric{T, CuSparseMatrixCSR{T, Cint}}, fact_alg::CUDSSFact, ::Type{T}) where {T} =
  if fact_alg.structure == "S"
    F = ldlt(K; view=fact_alg.view)
    type = :ldlt
  elseif fact_alg.structure == "SPD"
    F = cholesky(K; view=fact_alg.view)
    type = :cholesky
  else
    error("The provided structure is not supported by cuDSS.")
  end
  CUDSSFactorization(F, type, false, true)

init_fact(K::Symmetric{T, CuSparseMatrixCSR{T, Cint}}, fact_alg::CUDSSFact) where {T} =
  if fact_alg.structure == "S"
    F = ldlt(K; view=fact_alg.view)
    type = :ldlt
  elseif fact_alg.structure == "SPD"
    F = cholesky(K; view=fact_alg.view)
    type = :cholesky
  else
    error("The provided structure is not supported by cuDSS.")
  end
  CUDSSFactorization(F, type, false, true)

function generic_factorize!(
  K::Symmetric{T, CuSparseMatrixCSR{T, Cint}},
  K_fact::CUDSSFactorization{T},
) where {T}
  if K_fact.initialized
    K_fact.facCUDSSFactorizationtorized = false
    try
      (type == :ldlt) && ldlt!(K_fact.F, K.data)
      (type == :cholesky) && cholesky!(K_fact.F, K.data)
      K_fact.factorized = true
    catch
      K_fact.factorized = false
    end
  end
  !K_fact.initialized && (K_fact.initialized = true)
end

RipQP.factorized(K_fact::CUDSSFactorization) = K_fact.factorized

function ldiv!(K_fact::CUDSSFactorization, dd::CuVector)
  ldiv!(K_fact.F, dd)
end
