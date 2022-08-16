get_mat_QPData(
  A::SparseMatrixCOO{T, Int},
  H::SparseMatrixCOO{T, Int},
  nvar::Int,
  ncon::Int,
  sp::K2LDLParams{T0, F},
) where {T, T0, F <: HSLMA57Fact} = A, Symmetric(H, sp.uplo)

get_uplo(fact_alg::HSLMA57Fact) = :L

init_fact(K::Symmetric{T, SparseMatrixCOO{T, Int}}, fact_alg::HSLMA57Fact) where {T} =
  ma57_coord(K.data.rows, K.data.cols, K.data.vals, sqd = true)

function generic_factorize!(K::Symmetric{T, SparseMatrixCOO{T, Int}}, K_fact::Ma57) where {T}
  K_fact.vals .= K.data.vals
  ma57_factorize(K_fact)
end

import LinearAlgebra.ldiv!
ldiv!(K_fact::Ma57, dd::AbstractVector) = ma57_solve!(K_fact, dd)