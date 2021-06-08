struct IntDataInit{I <: Integer}
  nvar::I
  ncon::I
  ilow::Vector{I}
  iupp::Vector{I}
  irng::Vector{I}
  ifix::Vector{I}
  jlow::Vector{I}
  jupp::Vector{I}
  jrng::Vector{I}
  jfix::Vector{I}
end

function get_multipliers(
  s_l::AbstractVector{T},
  s_u::AbstractVector{T},
  ilow::Vector{Int},
  iupp::Vector{Int},
  nvar::Int,
  y::AbstractVector{T},
  idi::IntDataInit{Int},
) where {T <: Real}
  nlow, nupp, nrng = length(idi.ilow), length(idi.iupp), length(idi.irng)
  njlow, njupp, njrng = length(idi.jlow), length(idi.jupp), length(idi.jrng)

  s_l_sp = SparseVector(nvar, ilow, s_l)
  s_u_sp = SparseVector(nvar, iupp, s_u)

  if idi.nvar != nvar
    multipliers_L = s_l_sp[1:(idi.nvar)]
    multipliers_U = s_u_sp[1:(idi.nvar)]
  else
    multipliers_L = s_l_sp
    multipliers_U = s_u_sp
  end

  S = typeof(y)
  multipliers = fill!(S(undef, idi.ncon), zero(T))
  multipliers[idi.jfix] .= @views y[idi.jfix]
  multipliers[idi.jlow] .+= @views s_l[(nlow + nrng + 1):(nlow + nrng + njlow)]
  multipliers[idi.jupp] .-= @views s_u[(nupp + nrng + 1):(nupp + nrng + njupp)]
  multipliers[idi.jrng] .+=
    @views s_l[(nlow + nrng + njlow + 1):end] .- s_u[(nupp + nrng + njupp + 1):end]

  return multipliers, multipliers_L, multipliers_U
end
