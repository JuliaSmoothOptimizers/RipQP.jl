change_vector_eltype(S0::Type{<:Vector}, T) = S0.name.wrapper{T, 1}

convert_mat(M::Union{SparseMatrixCOO, SparseMatrixCSC}, T) =
  convert(typeof(M).name.wrapper{T, Int}, M)
convert_mat(M::Matrix, T) = convert(Matrix{T}, M)

function push_history_residuals!(
  res::ResidualsHistory{T},
  itd::IterData{T},
  pad::PreallocatedData{T},
  id::QM_IntData,
) where {T <: Real}
  push!(res.rbNormH, res.rbNorm)
  push!(res.rcNormH, res.rcNorm)
  push!(res.pddH, itd.pdd)
  push!(res.μH, itd.μ)

  bound_dist = zero(T)
  if id.nlow > 0 && id.nupp > 0
    bound_dist = min(minimum(itd.x_m_lvar), minimum(itd.uvar_m_x))
  elseif id.nlow > 0 && id.nupp == 0
    bound_dist = min(minimum(itd.x_m_lvar))
  elseif id.nlow == 0 && id.nupp > 0
    bound_dist = min(minimum(itd.uvar_m_x))
  end
  (id.nlow > 0 || id.nupp > 0) && push!(res.min_bound_distH, bound_dist)

  pad_type = typeof(pad)
  if pad_type <: PreallocatedDataAugmentedKrylov || pad_type <: PreallocatedDataNewtonKrylov
    push!(res.kiterH, niterations(pad.KS))
    push!(res.KresNormH, norm(res.Kres))
    push!(res.KresPNormH, @views norm(res.Kres[(id.nvar + 1):(id.nvar + id.ncon)]))
    push!(res.KresDNormH, @views norm(res.Kres[1:(id.nvar)]))
  elseif pad_type <: PreallocatedDataNormalKrylov
    push!(res.kiterH, niterations(pad.KS))
    push!(res.KresNormH, norm(res.Kres))
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

uses_krylov(pad::PreallocatedData) = false

# logs
function setup_log_header(pad::PreallocatedData{T}) where {T}
  if uses_krylov(pad)
    @info log_header(
      [:k, :pri_obj, :pdd, :rbNorm, :rcNorm, :α_pri, :α_du, :μ, :ρ, :δ, :kiter, :x],
      [Int, T, T, T, T, T, T, T, T, T, Int, Char],
      hdr_override = Dict(
        :k => "iter",
        :pri_obj => "obj",
        :pdd => "rgap",
        :rbNorm => "‖rb‖",
        :rcNorm => "‖rc‖",
      ),
    )
  else
    @info log_header(
      [:k, :pri_obj, :pdd, :rbNorm, :rcNorm, :α_pri, :α_du, :μ, :ρ, :δ],
      [Int, T, T, T, T, T, T, T, T, T],
      hdr_override = Dict(
        :k => "iter",
        :pri_obj => "obj",
        :pdd => "rgap",
        :rbNorm => "‖rb‖",
        :rcNorm => "‖rc‖",
      ),
    )
  end
end

function status_to_char(st::String)
  if st == "user-requested exit"
    return 'u'
  elseif st == "maximum number of iterations exceeded"
    return 'i'
  else
    return 's'
  end
end

function show_log_row_krylov(
  pad::PreallocatedData{T},
  itd::IterData{T},
  res::AbstractResiduals{T},
  cnts::Counters,
  α_pri::T,
  α_dual::T,
) where {T}
  @info log_row(
    Any[
      cnts.k,
      itd.minimize ? itd.pri_obj : -itd.pri_obj,
      itd.pdd,
      res.rbNorm,
      res.rcNorm,
      α_pri,
      α_dual,
      itd.μ,
      pad.regu.ρ,
      pad.regu.δ,
      niterations(pad.KS),
      status_to_char(pad.KS.stats.status),
    ],
  )
end

function show_log_row(
  pad::PreallocatedData{T},
  itd::IterData{T},
  res::AbstractResiduals{T},
  cnts::Counters,
  α_pri::T,
  α_dual::T,
) where {T}
  if uses_krylov(pad)
    show_log_row_krylov(pad, itd, res, cnts, α_pri, α_dual)
  else
    @info log_row(
      Any[
        cnts.k,
        itd.minimize ? itd.pri_obj : -itd.pri_obj,
        itd.pdd,
        res.rbNorm,
        res.rcNorm,
        α_pri,
        α_dual,
        itd.μ,
        pad.regu.ρ,
        pad.regu.δ,
      ],
    )
  end
end

solver_name(pad::PreallocatedData) = string(typeof(pad).name.name)[17:end]

function show_used_solver(pad::PreallocatedData{T}) where {T}
  slv_name = solver_name(pad)
  @info "Solving in $T using $slv_name"
end
