convert_sym(::Type{T}, Q::Symmetric{T0, M0}) where {T, T0, M0 <: AbstractMatrix{T0}} =
  Symmetric(convert_mat(Q.data, T), Symbol(Q.uplo))

function convert_FloatData(
  ::Type{T},
  fd_T0::QM_FloatData{T0, S0, M10, M20},
) where {T <: Real, T0 <: Real, S0, M10, M20}
  if T == T0
    return fd_T0
  else
    return QM_FloatData(
      convert_sym(T, fd_T0.Q),
      convert_mat(fd_T0.A, T),
      convert(change_vector_eltype(S0, T), fd_T0.b),
      convert(change_vector_eltype(S0, T), fd_T0.c),
      T(fd_T0.c0),
      convert(change_vector_eltype(S0, T), fd_T0.lvar),
      convert(change_vector_eltype(S0, T), fd_T0.uvar),
      fd_T0.uplo,
    )
  end
end

function convert_types(
  ::Type{T},
  pt::Point{T_old, S_old},
  itd::IterData{T_old, S_old},
  res::AbstractResiduals{T_old, S_old},
  dda::DescentDirectionAllocs{T_old, S_old},
  pad::PreallocatedData{T_old},
  sp_old::Union{Nothing, SolverParams},
  sp_new::Union{Nothing, SolverParams},
  id::QM_IntData,
  fd::Abstract_QM_FloatData, # type T
  solve_method_old::SolveMethod,
  solve_method_new::SolveMethod,
  ::Type{T0},
) where {T<: Real, T_old <: Real, S_old, T0 <: Real}
  S = S_old.name.wrapper{T, 1}
  (T == T_old) && (
    return pt,
    itd,
    res,
    convert_solve_method(DescentDirectionAllocs{T, S}, dda, solve_method_old, solve_method_new, id),
    convertpad(PreallocatedData{T}, pad, sp_old, sp_new, id, fd, T0)
  )
  pt = convert(Point{T, S}, pt)
  res = convert(AbstractResiduals{T, S}, res)
  itd = convert(IterData{T, S}, itd)
  pad = convertpad(PreallocatedData{T}, pad, sp_old, sp_new, id, fd, T0)
  dda = convert(DescentDirectionAllocs{T, S}, dda)
  dda =
    convert_solve_method(DescentDirectionAllocs{T, S}, dda, solve_method_old, solve_method_new, id)
  return pt, itd, res, dda, pad
end

small_αs(α_pri::T, α_dual::T, cnts::Counters) where {T} =
  (cnts.k ≥ 5) && ((α_pri < T(1.0e-1)) || (α_dual < T(1.0e-1)))
