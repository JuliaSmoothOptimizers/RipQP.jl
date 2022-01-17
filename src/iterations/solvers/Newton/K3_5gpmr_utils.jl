function ld_solve!(n, b::AbstractVector, Lp, Li, Lx, D, P)
  @views y = b[P]
  LDLFactorizations.ldl_lsolve!(n, y, Lp, Li, Lx)
  LDLFactorizations.ldl_dsolve!(n, y, D)
  return b
end

function dlt_solve!(n, b::AbstractVector, Lp, Li, Lx, D, P)
  @views y = b[P]
  LDLFactorizations.ldl_dsolve!(n, y, D)
  LDLFactorizations.ldl_ltsolve!(n, y, Lp, Li, Lx)
  return b
end

function ld_div!(
  y::AbstractVector{T},
  LDL::LDLFactorizations.LDLFactorization{Tf, Ti, Tn, Tp},
  b::AbstractVector{T},
) where {T <: Real, Tf <: Real, Ti <: Integer, Tn <: Integer, Tp <: Integer}
  y .= b
  LDL.__factorized || throw(LDLFactorizations.SQDException(error_string))
  ld_solve!(LDL.n, y, LDL.Lp, LDL.Li, LDL.Lx, LDL.d, LDL.P)
end

function dlt_div!(
  y::AbstractVector{T},
  LDL::LDLFactorizations.LDLFactorization{Tf, Ti, Tn, Tp},
  b::AbstractVector{T},
) where {T <: Real, Tf <: Real, Ti <: Integer, Tn <: Integer, Tp <: Integer}
  y .= b
  LDL.__factorized || throw(LDLFactorizations.SQDException(error_string))
  dlt_solve!(LDL.n, y, LDL.Lp, LDL.Li, LDL.Lx, LDL.d, LDL.P)
end

function opsqrtBRprod!(
  res::AbstractVector{T},
  ncon::Int,
  nlow::Int,
  x_m_lvar::AbstractVector{T},
  uvar_m_x::AbstractVector{T},
  δv::AbstractVector{T},
  v::AbstractVector{T},
  α::T,
  β::T,
) where {T <: Real}
  if β == zero(T)
    res[1:ncon] .= @views (α / sqrt.(δv[1])) .* v[1:ncon]
    res[(ncon + 1):(ncon + nlow)] .= @views α ./ sqrt.(x_m_lvar) .* v[(ncon + 1):(ncon + nlow)]
    res[(ncon + nlow + 1):end] .= @views α ./ sqrt.(uvar_m_x) .* v[(ncon + nlow + 1):end]
  else
    res[1:ncon] .= @views (α / sqrt.(δv[1])) .* v[1:ncon] .+ β .* res[1:ncon]
    res[(ncon + 1):(ncon + nlow)] .= @views α ./ sqrt.(x_m_lvar) .* v[(ncon + 1):(ncon + nlow)] .+
           β .* res[(ncon + 1):(ncon + nlow)]
    res[(ncon + nlow + 1):end] .=
      @views α ./ sqrt.(uvar_m_x) .* v[(ncon + nlow + 1):end] .+ β .* res[(ncon + nlow + 1):end]
  end
end
