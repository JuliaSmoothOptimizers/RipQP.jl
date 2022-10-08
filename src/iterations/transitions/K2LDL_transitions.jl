function convertpad(
  ::Type{<:PreallocatedData{T}},
  pad::PreallocatedDataK2LDL{T_old},
  sp_old::K2LDLParams,
  sp_new::Union{Nothing, K2LDLParams},
  id::QM_IntData,
  fd::Abstract_QM_FloatData,
  ::Type{T0},
) where {T <: Real, T_old <: Real, T0 <: Real}
  pad = PreallocatedDataK2LDL(
    convert(Array{T}, pad.D),
    convert(Regularization{T}, pad.regu),
    convert(SparseVector{T, Int}, pad.diag_Q),
    Symmetric(convert_mat(pad.K.data, T), Symbol(pad.K.uplo)),
    convertldl(T, pad.K_fact),
    pad.fact_fail,
    pad.diagind_K,
  )

  if pad.regu.regul == :classic
    if T == Float64 && typeof(sp_new) == Nothing
      pad.regu.ρ_min = 1e-5 * sqrt(eps())
      pad.regu.δ_min = 1e0 * sqrt(eps())
    else
      pad.regu.ρ_min, pad.regu.δ_min = T(sqrt(eps(T)) * 1e1), T(sqrt(eps(T)) * 1e1)
    end
  elseif pad.regu.regul == :dynamic
    pad.regu.ρ, pad.regu.δ = T(eps(T)^(3 / 4)), T(eps(T)^(0.45))
    pad.K_fact.LDL.r1, pad.K_fact.LDL.r2 = -pad.regu.ρ, pad.regu.δ
  end

  return pad
end
