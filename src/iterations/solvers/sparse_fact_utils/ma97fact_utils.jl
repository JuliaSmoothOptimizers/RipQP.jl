get_uplo(fact_alg::HSLMA97Fact) = :L

mutable struct Ma97Factorization{T} <: FactorizationData{T}
  ma97::Ma97{T}
end

function init_fact(
  K::Symmetric{T, SparseMatrixCSC{T, Int}},
  fact_alg::HSLMA97Fact,
  ::Type{T},
) where {T}
  return Ma97Factorization(
    ma97_csc(size(K, 1), Int32.(K.data.colptr), Int32.(K.data.rowval), K.data.nzval),
  )
end

function init_fact(K::Symmetric{T, SparseMatrixCSC{T, Int}}, fact_alg::HSLMA97Fact) where {T}
  return Ma97Factorization(
    ma97_csc(size(K, 1), Int32.(K.data.colptr), Int32.(K.data.rowval), K.data.nzval),
  )
end

function generic_factorize!(
  K::Symmetric{T, SparseMatrixCSC{T, Int}},
  K_fact::Ma97Factorization,
) where {T}
  copyto!(K_fact.ma97.nzval, K.data.nzval)
  ma97_factorize!(K_fact.ma97)
end

RipQP.factorized(K_fact::Ma97Factorization) = (K_fact.ma97.info.flag == 0)

ldiv!(K_fact::Ma97Factorization{T}, dd::AbstractVector{T}) where {T} = ma97_solve!(K_fact.ma97, dd)

function abs_diagonal!(K_fact::Ma97Factorization)
  K_fact
end

# function convertldl(T::DataType, K_fact::Ma97Factorization)
#   K_fact2 = Ma97Factorization(
#     Ma97{T, T}(
#       K_fact.ma97.__akeep,
#       K_fact.ma97.__fkeep,
#       K_fact.ma97.n,
#       K_fact.ma97.colptr,
#       K_fact.ma97.rowval,
#       convert(Array{T}, K_fact.ma97.nzval),
#       Ma97_Control{T}(
#         K_fact.ma97.control.f_arrays,
#         K_fact.ma97.control.action,
#         K_fact.ma97.control.nemin,
#         T(K_fact.ma97.control.multiplier),
#         K_fact.ma97.control.ordering,
#         K_fact.ma97.control.print_level,
#         K_fact.ma97.control.scaling,
#         T(K_fact.ma97.control.small),
#         T(K_fact.ma97.control.u),
#         K_fact.ma97.control.unit_diagnostics,
#         K_fact.ma97.control.unit_error,
#         K_fact.ma97.control.unit_warning,
#         K_fact.ma97.control.factor_min,
#         K_fact.ma97.control.solve_blas3,
#         K_fact.ma97.control.solve_min,
#         K_fact.ma97.control.solve_mf,
#         T(K_fact.ma97.control.consist_tol),
#         K_fact.ma97.control.ispare,
#         convert(Array{T}, K_fact.ma97.control.rspare),
#       ),
#       Ma97_Info{T}(
#         K_fact.ma97.info.flag,
#         K_fact.ma97.info.flag68,
#         K_fact.ma97.info.flag77,
#         K_fact.ma97.info.matrix_dup,
#         K_fact.ma97.info.matrix_rank,
#         K_fact.ma97.info.matrix_outrange,
#         K_fact.ma97.info.matrix_missing_diag,
#         K_fact.ma97.info.maxdepth,
#         K_fact.ma97.info.maxfront,
#         K_fact.ma97.info.num_delay,
#         K_fact.ma97.info.num_factor,
#         K_fact.ma97.info.num_flops,
#         K_fact.ma97.info.num_neg,
#         K_fact.ma97.info.num_sup,
#         K_fact.ma97.info.num_two,
#         K_fact.ma97.info.ordering,
#         K_fact.ma97.info.stat,
#         K_fact.ma97.info.ispare,
#         convert(Array{T}, K_fact.ma97.info.rspare),
#       ),
#     ),
#   )
#   HSL.ma97_finalize(K_fact2.ma97)
#   return K_fact2
# end
