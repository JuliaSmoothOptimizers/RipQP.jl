using .HSL

get_mat_QPData(
  A::SparseMatrixCOO{T, Int},
  H::SparseMatrixCOO{T, Int},
  nvar::Int,
  ncon::Int,
  sp::K2LDLParams{T0, F},
) where {T, T0, F <: HSLMA57Fact} = A, Symmetric(H, sp.uplo)

get_uplo(fact_alg::HSLMA57Fact) = :L

mutable struct Ma57Factorization{T}
  ma57::Ma57{T}
  work::Vector{T}
end

init_fact(K::Symmetric{T, SparseMatrixCOO{T, Int}}, fact_alg::HSLMA57Fact) where {T} =
  Ma57Factorization(ma57_coord(size(K, 1), K.data.rows, K.data.cols, K.data.vals, sqd = true), Vector{T}(undef, size(K, 1)))

function generic_factorize!(K::Symmetric{T, SparseMatrixCOO{T, Int}}, K_fact::Ma57Factorization{T}) where {T}
  K_fact.ma57.vals .= K.data.vals
  ma57_factorize!(K_fact.ma57)
end

LDLFactorizations.factorized(K_fact::Ma57Factorization) = true

import LinearAlgebra.ldiv!
ldiv!(K_fact::Ma57Factorization, dd::AbstractVector) = ma57_solve!(K_fact.ma57, dd, K_fact.work)
