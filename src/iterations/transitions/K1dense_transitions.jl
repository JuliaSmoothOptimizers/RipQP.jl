function convertpad(
  ::Type{<:PreallocatedData{T}},
  pad::PreallocatedDataK1CholDense{T0, S0, M0},
  sp_old::K1CholDenseParams,
  sp_new::Union{Nothing, K1CholDenseParams},
  id::QM_IntData,
  fd::Abstract_QM_FloatData,
  ::Type{T02}
) where {T <: Real, T0 <: Real, T02 <: Real, S0, M0}
  S = change_vector_eltype(S0, T)
  pad = PreallocatedDataK1CholDense(
    convert(S, pad.D),
    Diagonal(convert(S, pad.invD.diag)),
    convert_mat(pad.AinvD, T),
    convert(S, pad.rhs),
    convert(Regularization{T}, pad.regu),
    convert_mat(pad.K, T),
    pad.diagindK,
    convert(S, pad.tmpldiv),
  )

  if T == Float64 && T0 == Float64
    pad.regu.ρ_min, pad.regu.δ_min = T(sqrt(eps()) * 1e0), T(sqrt(eps()) * 1e0)
  else
    pad.regu.ρ_min, pad.regu.δ_min = T(sqrt(eps(T)) * 1e1), T(sqrt(eps(T)) * 1e1)
  end

  return pad
end
