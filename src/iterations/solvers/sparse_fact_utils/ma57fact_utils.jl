get_uplo(fact_alg::HSLMA57Fact) = :L

mutable struct Ma57Factorization{T} <: FactorizationData{T}
  ma57::Ma57{T}
  work::Vector{T}
end

function init_fact(K::Symmetric{T, SparseMatrixCOO{T, Int}}, fact_alg::HSLMA57Fact) where {T}
  K_fact = Ma57Factorization(
    ma57_coord(size(K, 1), K.data.rows, K.data.cols, K.data.vals, sqd = fact_alg.sqd),
    Vector{T}(undef, size(K, 1)),
  )
  return K_fact
end

function init_fact(
  K::Symmetric{T, SparseMatrixCOO{T, Int}},
  fact_alg::HSLMA57Fact,
  ::Type{Tf},
) where {T, Tf}
  K_fact = Ma57Factorization(
    ma57_coord(size(K, 1), K.data.rows, K.data.cols, Tf.(K.data.vals), sqd = fact_alg.sqd),
    Vector{Tf}(undef, size(K, 1)),
  )
  return K_fact
end

function generic_factorize!(
  K::Symmetric{T, SparseMatrixCOO{T, Int}},
  K_fact::Ma57Factorization,
) where {T}
  copyto!(K_fact.ma57.vals, K.data.vals)
  ma57_factorize!(K_fact.ma57)
end

RipQP.factorized(K_fact::Ma57Factorization) = (K_fact.ma57.info.info[1] == 0)

ldiv!(K_fact::Ma57Factorization{T}, dd::AbstractVector{T}) where {T} =
  ma57_solve!(K_fact.ma57, dd, K_fact.work)

function abs_diagonal!(K_fact::Ma57Factorization)
  K_fact
end

convertldl(T::DataType, K_fact::Ma57Factorization) = Ma57Factorization(
  Ma57(
    K_fact.ma57.n,
    K_fact.ma57.nz,
    K_fact.ma57.rows,
    K_fact.ma57.cols,
    convert(Array{T}, K_fact.ma57.vals),
    Ma57_Control(K_fact.ma57.control.icntl, convert(Array{T}, K_fact.ma57.control.cntl)),
    Ma57_Info(
      K_fact.ma57.info.info,
      convert(Array{T}, K_fact.ma57.info.rinfo),
      K_fact.ma57.info.largest_front,
      K_fact.ma57.info.num_2x2_pivots,
      K_fact.ma57.info.num_delayed_pivots,
      K_fact.ma57.info.num_negative_eigs,
      K_fact.ma57.info.rank,
      K_fact.ma57.info.num_pivot_sign_changes,
      T(K_fact.ma57.info.backward_error1),
      T(K_fact.ma57.info.backward_error2),
      T(K_fact.ma57.info.matrix_inf_norm),
      T(K_fact.ma57.info.solution_inf_norm),
      T(K_fact.ma57.info.scaled_residuals),
      T(K_fact.ma57.info.cond1),
      T(K_fact.ma57.info.cond2),
      T(K_fact.ma57.info.error_inf_norm),
    ),
    T(K_fact.ma57.multiplier),
    K_fact.ma57.__lkeep,
    K_fact.ma57.__keep,
    K_fact.ma57.__lfact,
    convert(Array{T}, K_fact.ma57.__fact),
    K_fact.ma57.__lifact,
    K_fact.ma57.__ifact,
    K_fact.ma57.iwork_fact,
    K_fact.ma57.iwork_solve,
  ),
  convert(Array{T}, K_fact.work),
)
