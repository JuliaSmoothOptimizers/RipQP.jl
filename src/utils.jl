function push_history_residuals!(res::Residuals{T}, pdd::T, pad::PreallocatedData) where {T <: Real}
  push!(res.rbNormH, res.rbNorm)
  push!(res.rcNormH, res.rcNorm)
  push!(res.pddH, pdd)
  pad_type = typeof(pad)
  if (pad_type <: PreallocatedData_K2Krylov || pad_type <: PreallocatedData_K2_5Krylov)
    push!(res.nprodH, pad.K.nprod)
    pad.K.nprod = 0
  end
end

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

  S = typeof(y)
  if S <: Vector
    s_l_sp = SparseVector(nvar, ilow, s_l)
    s_u_sp = SparseVector(nvar, iupp, s_u)
  else
    s_l_sp, s_u_sp = fill!(S(undef, nvar), zero(T)), fill!(S(undef, nvar), zero(T))
    s_l_sp[ilow] .= s_l
    s_u_sp[iupp] .= s_u
  end

  if idi.nvar != nvar
    multipliers_L = s_l_sp[1:(idi.nvar)]
    multipliers_U = s_u_sp[1:(idi.nvar)]
  else
    multipliers_L = s_l_sp
    multipliers_U = s_u_sp
  end

  multipliers = fill!(S(undef, idi.ncon), zero(T))
  multipliers[idi.jfix] .= @views y[idi.jfix]
  multipliers[idi.jlow] .+= @views s_l[(nlow + nrng + 1):(nlow + nrng + njlow)]
  multipliers[idi.jupp] .-= @views s_u[(nupp + nrng + 1):(nupp + nrng + njupp)]
  multipliers[idi.jrng] .+=
    @views s_l[(nlow + nrng + njlow + 1):end] .- s_u[(nupp + nrng + njupp + 1):end]

  return multipliers, multipliers_L, multipliers_U
end
