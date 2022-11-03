change_vector_eltype(S0::Type{<:Vector}, ::Type{T}) where {T} = S0.name.wrapper{T, 1}

convert_mat(M::Union{SparseMatrixCOO, SparseMatrixCSC}, ::Type{T}) where {T} =
  convert(typeof(M).name.wrapper{T, Int}, M)
convert_mat(M::Matrix, ::Type{T}) where {T} = convert(Matrix{T}, M)

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
    push!(res.kiterH, pad.kiter)
    push!(res.KresNormH, norm(res.Kres))
    push!(res.KresPNormH, @views norm(res.Kres[(id.nvar + 1):(id.nvar + id.ncon)]))
    push!(res.KresDNormH, @views norm(res.Kres[1:(id.nvar)]))
  elseif pad_type <: PreallocatedDataNormalKrylov
    push!(res.kiterH, niterations(pad.KS))
    push!(res.KresNormH, norm(res.Kres))
  end
end

function get_slack_multipliers(
  multipliers_in::AbstractVector{T},
  multipliers_L_in::AbstractVector{T},
  multipliers_U_in::AbstractVector{T},
  id::QM_IntData,
  idi::IntDataInit{Int},
) where {T <: Real}
  nlow, nupp, nrng = length(idi.ilow), length(idi.iupp), length(idi.irng)
  njlow, njupp, njrng = length(idi.jlow), length(idi.jupp), length(idi.jrng)

  S = typeof(multipliers_in)
  if idi.nvar != id.nvar
    multipliers_L = multipliers_L_in[1:(idi.nvar)]
    multipliers_U = multipliers_U_in[1:(idi.nvar)]
  else
    multipliers_L = multipliers_L_in
    multipliers_U = multipliers_U_in
  end

  multipliers = fill!(S(undef, idi.ncon), zero(T))
  multipliers[idi.jfix] .= @views multipliers_in[idi.jfix]
  multipliers[idi.jlow] .+= @views multipliers_L_in[id.ilow[(nlow + nrng + 1):(nlow + nrng + njlow)]]
  multipliers[idi.jupp] .-= @views multipliers_U_in[id.iupp[(nupp + nrng + 1):(nupp + nrng + njupp)]]
  multipliers[idi.jrng] .+=
    @views multipliers_L_in[id.ilow[(nlow + nrng + njlow + 1):end]] .-
      multipliers_U_in[id.iupp[(nupp + nrng + njupp + 1):end]]

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

status_to_char(KS::KrylovSolver) = status_to_char(KS.stats.status)

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
      pad.kiter,
      status_to_char(pad.KS),
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

solver_type(sp::SolverParams{T}) where {T} = T

function next_type(::Type{T}, ::Type{T0}) where {T, T0}
  T == T0 && return T0
  T == Float32 && return Float64
  T == Float64 && return T0
end

function set_ripqp_solver_specific!(stats::GenericExecutionStats, field::Symbol, value)
  stats.solver_specific[field] = value
end

function set_ripqp_bounds_multipliers!(
  stats::GenericExecutionStats,
  s_l::AbstractVector{T},
  s_u::AbstractVector{T},
  ilow::AbstractVector{Int},
  iupp::AbstractVector{Int},
) where {T}
  stats.multipliers_L[ilow] .= s_l
  stats.multipliers_U[iupp] .= s_u
end

function ripqp_solver_specific(QM::AbstractQuadraticModel{T}, history::Bool) where {T}
  if history
    solver_specific = Dict(
      :relative_iter_cnt => -1,
      :iters_sp => -1,
      :iters_sp2 => -1,
      :iters_sp3 => -1,
      :pdd => T(Inf),
      :psoperations => (typeof(QM) <: QuadraticModels.PresolvedQuadraticModel) ? QM.psd.operations : [],
      :rbNormH => T[],
      :rcNormH => T[],
      :pddH => T[],
      :nprodH => Int[],
      :min_bound_distH => T[],
      :KresNormH => T[],
      :KresPNormH => T[],
      :KresDNormH => T[],
    )
  else
    solver_specific = Dict(
      :relative_iter_cnt => -1,
      :iters_sp => -1,
      :iters_sp2 => -1,
      :iters_sp3 => -1,
      :pdd => T(Inf),
      :psoperations => (typeof(QM) <: QuadraticModels.PresolvedQuadraticModel) ? QM.psd.operations : [],
    )
  end
  return solver_specific
end

function set_ripqp_stats!(
  stats::GenericExecutionStats,
  pt::Point{T},
  res::AbstractResiduals{T},
  pad::PreallocatedData{T},
  itd::IterData{T},
  id::QM_IntData,
  sc::StopCrit,
  cnts::Counters,
  max_iter::Int,
) where {T}

  if cnts.k >= max_iter
    status = :max_iter
  elseif sc.tired
    status = :max_time
  elseif sc.optimal
    status = :first_order
  else
    status = :unknown
  end
  set_status!(stats, status)
  set_solution!(stats, pt.x)
  set_constraint_multipliers!(stats, pt.y)
  set_ripqp_bounds_multipliers!(stats, pt.s_l, pt.s_u, id.ilow, id.iupp)
  set_objective!(stats, itd.minimize ? itd.pri_obj : -itd.pri_obj)
  set_primal_residual!(stats, res.rbNorm)
  set_dual_residual!(stats, res.rcNorm)
  set_iter!(stats, cnts.k)
  elapsed_time = time() - cnts.time_solve
  set_time!(stats, elapsed_time)
  # set stats.solver_specific
  set_ripqp_solver_specific!(stats, :relative_iter_cnt, cnts.km)
  set_ripqp_solver_specific!(stats, :iters_sp, cnts.iters_sp)
  set_ripqp_solver_specific!(stats, :iters_sp2, cnts.iters_sp2)
  set_ripqp_solver_specific!(stats, :iters_sp3, cnts.iters_sp3)
  set_ripqp_solver_specific!(stats, :pdd, itd.pdd)
  if typeof(res) <: ResidualsHistory
    set_ripqp_solver_specific!(stats, :rbNormH, res.rbNormH)
    set_ripqp_solver_specific!(stats, :rcNormH, res.rcNormH)
    set_ripqp_solver_specific!(stats, :pddH, res.pddH)
    set_ripqp_solver_specific!(stats, :nprodH, res.kiterH)
    set_ripqp_solver_specific!(stats, :μH, res.μH)
    set_ripqp_solver_specific!(stats, :min_bound_distH, res.min_bound_distH)
    set_ripqp_solver_specific!(stats, :KresNormH, res.KresNormH)
    set_ripqp_solver_specific!(stats, :KresPNormH, res.KresPNormH)
    set_ripqp_solver_specific!(stats, :KresDNormH, res.KresDNormH)
  end
  if pad isa PreallocatedDataK2Krylov &&
      pad.pdat isa LDLData &&
      pad.pdat.K_fact isa LDLFactorizationData
    nnzLDL = length(pad.pdat.K_fact.LDL.Lx) + length(pad.pdat.K_fact.LDL.d)
    set_ripqp_solver_specific!(stats, :nnzLDL, nnzLDL)
  end
end

function get_inner_model_data(
  QM::QuadraticModel{T},
  QMps::AbstractQuadraticModel{T},
  ps::Bool,
  scaling::Bool,
  history::Bool,
) where {T <: Real}
  # save inital IntData to compute multipliers at the end of the algorithm
  idi = IntDataInit(QM)
  QM_inner = SlackModel(QMps)
  if QM_inner.meta.ncon == length(QM_inner.meta.jfix) && !ps && scaling
    QM_inner = deepcopy(QM_inner) # if not modified by SlackModel and presolve
  end
  if !QM.meta.minimize && !ps # switch to min problem if not modified by presolve
    QuadraticModels.switch_H_to_max!(QM_inner.data)
    QM_inner.data.c .= .-QM_inner.data.c
    QM_inner.data.c0 = -QM_inner.data.c0
  end
  stats_inner = GenericExecutionStats(
    QM_inner;
    solution = QM_inner.meta.x0,
    multipliers = QM_inner.meta.y0,
    multipliers_L = fill!(similar(QM_inner.meta.x0), zero(T)),
    multipliers_U = fill!(similar(QM_inner.meta.x0), zero(T)),
    solver_specific = ripqp_solver_specific(QM_inner, history),
  )
  return QM_inner, stats_inner, idi
end

function get_stats_outer(
  stats_inner::GenericExecutionStats{T},
  QM::QuadraticModel{T},
  QMps::AbstractQuadraticModel{T},
  id::QM_IntData,
  idi::IntDataInit,
  start_time::Float64,
  ps::Bool,
) where {T <: Real}
  multipliers, multipliers_L, multipliers_U = get_slack_multipliers(
    stats_inner.multipliers,
    stats_inner.multipliers_L,
    stats_inner.multipliers_U,
    id,
    idi,
  ) 
  if ps
    sol_in = QMSolution(stats_inner.solution, stats_inner.multipliers, stats_inner.multipliers_L, stats_inner.multipliers_U)
    sol = postsolve(QM, QMps, sol_in)
    x, multipliers, multipliers_L, multipliers_U = sol.x, sol.y, sol.s_l, sol.s_u
  else
    x = stats_inner.solution[1:(idi.nvar)]
  end
  solver_specific = stats_inner.solver_specific
  solver_specific[:psoperations] = QMps isa QuadraticModels.PresolvedQuadraticModel ? QMps.psd.operations : []
  elapsed_time = time() - start_time
  return GenericExecutionStats(
    QM,
    status = stats_inner.status,
    solution = x,
    objective = stats_inner.objective,
    dual_feas = stats_inner.dual_feas,
    primal_feas = stats_inner.primal_feas,
    multipliers = multipliers,
    multipliers_L = multipliers_L,
    multipliers_U = multipliers_U,
    iter = stats_inner.iter,
    elapsed_time = elapsed_time,
    solver_specific = solver_specific,
  )
end
