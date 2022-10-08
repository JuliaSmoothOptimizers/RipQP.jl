function convertpad(
  ::Type{<:PreallocatedData{T}},
  pad::PreallocatedDataK2_5LDL{T_old},
  sp_old::K2_5LDLParams,
  sp_new::Union{Nothing, K2_5LDLParams},
  id::QM_IntData,
  fd::Abstract_QM_FloatData,
  ::Type{T0},
) where {T <: Real, T_old <: Real, T0 <: Real}
  pad = PreallocatedDataK2_5LDL(
    convert(Array{T}, pad.D),
    convert(Regularization{T}, pad.regu),
    convert(SparseVector{T, Int}, pad.diag_Q),
    Symmetric(convert(SparseMatrixCSC{T, Int}, pad.K.data), Symbol(pad.K.uplo)),
    convertldl(T, pad.K_fact),
    pad.fact_fail,
    pad.diagind_K,
    pad.K_scaled,
  )

  if pad.regu.regul == :classic
    if T == Float64 && T0 == Float64
      pad.regu.ρ_min, pad.regu.δ_min = T(sqrt(eps()) * 1e-5), T(sqrt(eps()) * 1e0)
    else
      pad.regu.ρ_min, pad.regu.δ_min = T(sqrt(eps(T)) * 1e1), T(sqrt(eps(T)) * 1e1)
    end
    pad.regu.ρ /= 10
    pad.regu.δ /= 10
  elseif pad.regu.regul == :dynamic
    pad.regu.ρ, pad.regu.δ = T(eps(T)^(3 / 4)), T(eps(T)^(0.45))
    pad.K_fact.LDL.r1, pad.K_fact.LDL.r2 = -pad.regu.ρ, pad.regu.δ
  end

  return pad
end
