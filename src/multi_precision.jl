convert_sym(T::DataType, Q::Symmetric{T0, M0}) where {T0, M0 <: AbstractMatrix{T0}} =
  Symmetric(convert(M0.name.wrapper{T, Int}, Q.data), Symbol(Q.uplo))

function convert_FloatData(
  T::DataType,
  fd_T0::QM_FloatData{T0, S0, M10, M20},
) where {T0 <: Real, S0, M10, M20}
  return QM_FloatData(
    convert_sym(T, fd_T0.Q),
    convert(M20.name.wrapper{T, Int}, fd_T0.A),
    S0.name.wrapper{T, 1}(fd_T0.b),
    S0.name.wrapper{T, 1}(fd_T0.c),
    T(fd_T0.c0),
    S0.name.wrapper{T, 1}(fd_T0.lvar),
    S0.name.wrapper{T, 1}(fd_T0.uvar),
    fd_T0.uplo,
  )
end

function convert_types(
  T::DataType,
  pt::Point{T_old, S_old},
  itd::IterData{T_old, S_old},
  res::AbstractResiduals{T_old, S_old},
  dda::DescentDirectionAllocs{T_old, S_old},
  pad::PreallocatedData{T_old},
  T0::DataType,
) where {T_old <: Real, S_old}
  S = S_old.name.wrapper{T, 1}
  pt = convert(Point{T, S}, pt)
  res = convert(AbstractResiduals{T, S}, res)
  itd = convert(IterData{T, S}, itd)
  pad = convertpad(PreallocatedData{T}, pad, T0)
  dda = convert(DescentDirectionAllocs{T, S}, dda)
  return pt, itd, res, dda, pad
end

function iter_and_update_T!(
  pt::Point{T},
  itd::IterData{T},
  fd_T::Abstract_QM_FloatData{T},
  id::QM_IntData,
  res::AbstractResiduals{T},
  sc::StopCrit{Tsc},
  dda::DescentDirectionAllocs{T},
  pad::PreallocatedData{T},
  ϵ_T::Tolerances{T},
  ϵ::Tolerances{T0},
  cnts::Counters,
  max_iter_T::Int,
  T_next::DataType,
  display::Bool,
) where {T <: Real, T0 <: Real, Tsc <: Real}
  # iters T
  sc.max_iter = max_iter_T
  iter!(pt, itd, fd_T, id, res, sc, dda, pad, ϵ_T, cnts, T0, display)

  # convert to T_next
  pt, itd, res, dda, pad = convert_types(T_next, pt, itd, res, dda, pad, T0)
  sc.optimal = itd.pdd < ϵ.pdd && res.rbNorm < ϵ.tol_rb && res.rcNorm < ϵ.tol_rc
  sc.small_μ = itd.μ < ϵ.μ
  return pt, itd, res, dda, pad
end
