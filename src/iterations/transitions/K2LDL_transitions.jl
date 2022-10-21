function convertpad(
  ::Type{<:PreallocatedData{T}},
  pad::PreallocatedDataK2LDL{T_old},
  sp_old::K2LDLParams,
  sp_new::K2LDLParams,
  id::QM_IntData,
  fd::Abstract_QM_FloatData,
) where {T <: Real, T_old <: Real}
  pad = PreallocatedDataK2LDL(
    convert(Array{T}, pad.D),
    convert(Regularization{T}, pad.regu),
    sp_new.bypass_bound_dist_safety,
    convert(SparseVector{T, Int}, pad.diag_Q),
    Symmetric(convert_mat(pad.K.data, T), Symbol(pad.K.uplo)),
    convertldl(T, pad.K_fact),
    pad.fact_fail,
    pad.diagind_K,
  )

  if pad.regu.regul == :classic
      pad.regu.ρ_min, pad.regu.δ_min = T(sp_new.ρ_min), T(sp_new.ρ_min)
  elseif pad.regu.regul == :dynamic
    pad.regu.ρ, pad.regu.δ = T(eps(T)^(3 / 4)), T(eps(T)^(0.45))
    pad.K_fact.LDL.r1, pad.K_fact.LDL.r2 = -pad.regu.ρ, pad.regu.δ
  end

  return pad
end
