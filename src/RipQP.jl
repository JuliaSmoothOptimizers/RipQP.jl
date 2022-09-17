module RipQP

using DelimitedFiles, LinearAlgebra, MatrixMarket, Quadmath, SparseArrays, Statistics, TimerOutputs

using Krylov,
  LDLFactorizations,
  LimitedLDLFactorizations,
  LinearOperators,
  LLSModels,
  NLPModelsModifiers,
  QuadraticModels,
  SolverCore,
  SparseMatricesCOO

using Requires
function __init__()
  @require CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba" include("gpu_utils.jl")
  @require HSL = "34c5aeac-e683-54a6-a0e9-6e0fdc586c50" include(
    "iterations/solvers/sparse_fact_utils/hslfact_utils.jl",
  )
  @require QDLDL = "bfc457fd-c171-5ab7-bd9e-d5dbfc242d63" include(
    "iterations/solvers/sparse_fact_utils/qdldl_utils.jl",
  )
  @require SuiteSparse = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9" include(
    "iterations/solvers/sparse_fact_utils/cholmod_utils.jl",
  )
end

export ripqp

include("types_definition.jl")
include("iterations/iterations.jl")
include("refinement.jl")
include("data_initialization.jl")
include("starting_points.jl")
include("scaling.jl")
include("multi_precision.jl")
include("utils.jl")

const to = TimerOutput()

"""
    stats = ripqp(QM :: QuadraticModel{T0};
                  itol = InputTol(T0), scaling = true, ps = true,
                  normalize_rtol = true, kc = 0, mode = :mono, perturb = false,
                  early_multi_stop = true,
                  sp = (mode == :mono) ? K2LDLParams{T0}() : K2LDLParams{Float32}(),
                  sp2 = nothing, sp3 = nothing, 
                  solve_method = PC(), 
                  history = false, w = SystemWrite(), display = true) where {T0<:Real}

Minimize a convex quadratic problem. Algorithm stops when the criteria in pdd, rb, and rc are valid.
Returns a [GenericExecutionStats](https://juliasmoothoptimizers.github.io/SolverCore.jl/dev/reference/#SolverCore.GenericExecutionStats) 
containing information about the solved problem.

- `QM :: QuadraticModel`: problem to solve
- `itol :: InputTol{T, Int}` input Tolerances for the stopping criteria. See [`RipQP.InputTol`](@ref).
- `scaling :: Bool`: activate/deactivate scaling of A and Q in `QM0`
- `ps :: Bool` : activate/deactivate presolve
- `normalize_rtol :: Bool = true` : if `true`, the primal and dual tolerance for the stopping criteria 
    are normalized by the initial primal and dual residuals
- `kc :: Int`: number of centrality corrections (set to `-1` for automatic computation)
- `perturb :: Bool` : activate / deativate perturbation of the current point when μ is too small
- `mode :: Symbol`: should be `:mono` to use the mono-precision mode, `:multi` to use
    the multi-precision mode (start in single precision and gradually transitions
    to `T0`), `:zoom` to use the zoom procedure, `:multizoom` to use the zoom procedure 
    with multi-precision, `ref` to use the QP refinement procedure, or `multiref` 
    to use the QP refinement procedure with multi_precision
- `early_multi_stop :: Bool` : stop the iterations in lower precision systems earlier in multi-precision mode,
    based on some quantities of the algorithm
- `sp :: SolverParams` : choose a solver to solve linear systems that occurs at each iteration and during the 
    initialization, see [`RipQP.SolverParams`](@ref)
- `sp2 :: Union{Nothing, SolverParams}` and `sp3 :: Union{Nothing, SolverParams}` : choose second and third solvers
    to solve linear systems that occurs at each iteration in the second and third solving phase.
    When `mode != :mono`, leave to `nothing` if you want to keep using `sp`.
    If `sp2` is not nothing, then `mode` should be set to `:multi`, `:multiref` or `multizoom`.  
- `solve_method :: SolveMethod` : method used to solve the system at each iteration, use `solve_method = PC()` to 
    use the Predictor-Corrector algorithm (default), and use `solve_method = IPF()` to use the Infeasible Path 
    Following algorithm
- `history :: Bool` : set to true to return the primal and dual norm histories, the primal-dual relative difference
    history, and the number of products if using a Krylov method in the `solver_specific` field of the 
    [GenericExecutionStats](https://juliasmoothoptimizers.github.io/SolverCore.jl/dev/reference/#SolverCore.GenericExecutionStats)
- `w :: SystemWrite`: configure writing of the systems to solve (no writing is done by default), see [`RipQP.SystemWrite`](@ref)
- `display::Bool`: activate/deactivate iteration data display

You can also use `ripqp` to solve a [LLSModel](https://juliasmoothoptimizers.github.io/LLSModels.jl/stable/#LLSModels.LLSModel):

    stats = ripqp(LLS::LLSModel{T0}; mode = :mono,
                  sp = (mode == :mono) ? K2LDLParams{T0}() : K2LDLParams{Float32}(), 
                  kwargs...) where {T0 <: Real}
"""
function ripqp(
  QM0::QuadraticModel{T0};
  itol::InputTol{T0, Int} = InputTol(T0),
  scaling::Bool = true,
  ps::Bool = true,
  normalize_rtol::Bool = true,
  kc::I = 0,
  perturb::Bool = false,
  mode::Symbol = :mono,
  early_multi_stop::Bool = true,
  sp::SolverParams = (mode == :mono) ? K2LDLParams{T0}() : K2LDLParams{Float32}(),
  sp2::Union{Nothing, SolverParams} = nothing,
  sp3::Union{Nothing, SolverParams} = nothing,
  solve_method::SolveMethod = PC(),
  history::Bool = false,
  w::SystemWrite = SystemWrite(),
  display::Bool = true,
) where {T0 <: Real, I <: Integer}
  start_time = time()
  elapsed_time = 0.0
  @timeit_debug to "ripqp" begin
    # config
    mode == :mono ||
      mode == :multi ||
      mode == :zoom ||
      mode == :multizoom ||
      mode == :ref ||
      mode == :multiref ||
      error("not a valid mode")
    if !isnothing(sp2) && mode == :mono
      @warn "setting mode to multi because sp2 is provided"
      mode = :multi
    end
    typeof(solve_method) <: IPF &&
      kc != 0 &&
      error("IPF method should not be used with centrality corrections")
    Ti = solver_type(sp) # initial solve type
    iconf = InputConfig(
      mode,
      early_multi_stop,
      scaling,
      ps || (QM0.data.H isa SparseMatrixCOO && QM0.data.A isa SparseMatrixCOO),
      normalize_rtol,
      kc,
      perturb,
      QM0.meta.minimize,
      sp,
      sp2,
      sp3,
      solve_method,
      history,
      w,
    )

    # conversion function if QM.data.H and QM.data.A are not in the type required by iconf.sp
    QM0 = convert_QM(QM0, iconf, display)

    if iconf.presolve
      stats_ps = presolve(QM0, fixed_vars_only = !ps)
      if stats_ps.status == :unknown
        QM = stats_ps.solver_specific[:presolvedQM]
      else
        return stats_ps
      end
    else
      QM = QM0
    end

    # allocate workspace
    sc, idi, fd_T0, id, ϵ_T0, res, itd, dda, pt, sd, spd, cnts, T =
      @timeit_debug to "allocate workspace" allocate_workspace(QM, iconf, itol, start_time, T0, sp)

    if iconf.scaling
      scaling!(fd_T0, id, sd, T0(1.0e-5))
    end

    # extra workspace for multi mode
    if (iconf.mode == :multi || iconf.mode == :multizoom || iconf.mode == :multiref)
      T2 = next_type(Ti, T0) # eltype of sp2
      # if the 2nd solver is nothing:
      isnothing(sp2) && (sp2 = eval(typeof(sp).name.name){T2}())
      fd1, ϵ1 = allocate_extra_workspace1(Ti, itol, iconf, fd_T0)
      fd, ϵ = fd1, ϵ1
      if T2 != T0 || !isnothing(sp3)
        fd2, ϵ2 = allocate_extra_workspace2(T2, itol, iconf, fd_T0)
        # if the 3nd solver is nothing:
        isnothing(sp3) && (sp3 = eval(typeof(sp2).name.name){T0}())
      end
    elseif iconf.mode == :ref || iconf.mode == :zoom
      fd = fd_T0
      ϵ = Tolerances(
        T(1),
        T(itol.ϵ_rbz),
        T(itol.ϵ_rbz),
        T(ϵ_T0.tol_rb * T(itol.ϵ_rbz / itol.ϵ_rb)),
        one(T),
        T(itol.ϵ_μ),
        T(itol.ϵ_Δx),
        iconf.normalize_rtol,
      )
    else
      fd, ϵ = fd_T0, ϵ_T0
    end

    # initialize data (some allocations for the pad creation)
    pad = initialize!(fd, id, res, itd, dda, pt, spd, ϵ, sc, iconf, cnts, T0)
    if (iconf.mode == :multi || iconf.mode == :multizoom || iconf.mode == :multiref)
      # set intermediate tolerances
      set_tol_residuals!(ϵ_T0, T0(res.rbNorm), T0(res.rcNorm))
      !isnothing(sp3) && set_tol_residuals!(ϵ2, T2(res.rbNorm), T2(res.rcNorm))
    end

    Δt = time() - start_time
    sc.tired = Δt > itol.max_time

    # display
    if display == true
      @timeit_debug to "display" begin
        show_used_solver(pad)
        setup_log_header(pad)
        show_log_row(pad, itd, res, cnts, zero(T), zero(T))
      end
    end

    last_iter = iconf.mode == :mono # do not stop early in mono mode
    !isnothing(sp2) && (sc.max_iter = itol.max_iter1) # set max_iter for 1st solver
    # IPM iterations 1st solver (mono mode: only this call to the iter! function)
    iter!(pt, itd, fd, id, res, sc, dda, pad, ϵ, cnts, iconf, T0, display, last_iter = last_iter)

    # initialize iteration counters
    iters_sp, iters_sp2, iters_sp3 = cnts.k, 0, 0

    if !isnothing(sp2) # setup data for 2nd solver
      fd = isnothing(sp3) ? fd_T0 : fd2
      ϵ = isnothing(sp3) ? ϵ_T0 : ϵ2
      pt, itd, res, dda, pad = convert_types(T2, pt, itd, res, dda, pad, sp, sp2, id, fd, T0)
      sc.optimal = itd.pdd < ϵ_T0.pdd && res.rbNorm < ϵ_T0.tol_rb && res.rcNorm < ϵ_T0.tol_rc
      sc.small_μ = itd.μ < ϵ_T0.μ
      display && show_used_solver(pad)
      display && setup_log_header(pad)

      # set max_iter for 2nd solver
      sc.max_iter = isnothing(sp3) ? itol.max_iter : itol.max_iter2
    end

    if !isnothing(sp3) # iter! with 2nd solver, setup data 3rd solver
      # IPM iterations 2nd solver starting from pt
      iter!(
        pt,
        itd,
        fd,
        id,
        res,
        sc,
        dda,
        pad,
        ϵ,
        cnts,
        iconf,
        T0,
        display,
        last_iter = !isnothing(sp3),
      )

      # iteration counter for 2nd solver
      iters_sp2 = cnts.k - iters_sp

      fd = fd_T0
      ϵ = ϵ_T0
      pt, itd, res, dda, pad = convert_types(T0, pt, itd, res, dda, pad, sp2, sp3, id, fd, T0)
      sc.optimal = itd.pdd < ϵ_T0.pdd && res.rbNorm < ϵ_T0.tol_rb && res.rcNorm < ϵ_T0.tol_rc
      sc.small_μ = itd.μ < ϵ_T0.μ
      display && show_used_solver(pad)
      display && setup_log_header(pad)

      # set max_iter for 3rd solver
      sc.max_iter = itol.max_iter
    end

    # IPM iterations 3rd solver: different following the use of mode multi, multizoom, multiref
    # starting from pt
    if !sc.optimal && mode == :multi
      iter!(pt, itd, fd, id, res, sc, dda, pad, ϵ, cnts, iconf, T0, display)
    elseif !sc.optimal && (mode == :multizoom || mode == :multiref)
      spd = convert(StartingPointData{T0, typeof(pt.x)}, spd)
      fd_ref, pt_ref = fd_refinement(
        fd,
        id,
        res,
        itd.Δxy,
        pt,
        itd,
        ϵ,
        dda,
        pad,
        spd,
        cnts,
        T0,
        iconf.mode,
        centering = true,
      )
      iter!(pt_ref, itd, fd_ref, id, res, sc, dda, pad, ϵ, cnts, iconf, T0, display)
      update_pt_ref!(fd_ref.Δref, pt, pt_ref, res, id, fd_T0, itd)
    elseif iconf.mode == :zoom || iconf.mode == :ref
      ϵ = ϵ_T0
      sc.optimal = false
      fd_ref, pt_ref =
        fd_refinement(fd, id, res, itd.Δxy, pt, itd, ϵ, dda, pad, spd, cnts, T0, iconf.mode)
      iter!(pt_ref, itd, fd_ref, id, res, sc, dda, pad, ϵ, cnts, iconf, T0, display)
      update_pt_ref!(fd_ref.Δref, pt, pt_ref, res, id, fd_T0, itd)
    end

    # update number of iterations for each solver 
    if isnothing(sp3) && !isnothing(sp2)
      iters_sp2 = cnts.k - iters_sp
    elseif !isnothing(sp3)
      iters_sp3 = cnts.k - iters_sp2
    end

    if iconf.scaling
      post_scale!(sd, pt, res, fd_T0, id, itd)
    end

    if cnts.k >= itol.max_iter
      status = :max_iter
    elseif sc.tired
      status = :max_time
    elseif sc.optimal
      status = :first_order
    else
      status = :unknown
    end

    multipliers_in, multipliers_L, multipliers_U =
      get_multipliers(pt.s_l, pt.s_u, id.ilow, id.iupp, id.nvar, pt.y, idi)

    if iconf.presolve
      sol_in = QMSolution(pt.x, multipliers_in, multipliers_L, multipliers_U)
      sol = postsolve(QM0, QM, sol_in)
      x, multipliers, multipliers_L, multipliers_U = sol.x, sol.y, sol.s_l, sol.s_u
    else
      x = pt.x[1:(idi.nvar)]
      multipliers = pt.y
    end

    psoperations = typeof(QM) <: QuadraticModels.PresolvedQuadraticModel ? QM.psd.operations : []
    if typeof(res) <: ResidualsHistory
      solver_specific = Dict(
        :relative_iter_cnt => cnts.km,
        :iters_sp => iters_sp,
        :iters_sp2 => iters_sp2,
        :iters_sp3 => iters_sp3,
        :pdd => itd.pdd,
        :nvar_slack => id.nvar,
        :rbNormH => res.rbNormH,
        :rcNormH => res.rcNormH,
        :pddH => res.pddH,
        :nprodH => res.kiterH,
        :μH => res.μH,
        :min_bound_distH => res.min_bound_distH,
        :KresNormH => res.KresNormH,
        :KresPNormH => res.KresPNormH,
        :KresDNormH => res.KresDNormH,
        :psoperations => psoperations,
      )
    else
      solver_specific = Dict(
        :relative_iter_cnt => cnts.km,
        :iters_sp => iters_sp,
        :iters_sp2 => iters_sp2,
        :iters_sp3 => iters_sp3,
        :pdd => itd.pdd,
        :nvar_slack => id.nvar,
        :psoperations => psoperations,
      )
    end

    if pad isa PreallocatedDataK2Krylov &&
       pad.pdat isa LDLData &&
       pad.pdat.K_fact isa LDLFactorizationData
      solver_specific[:nnzLDL] = length(pad.pdat.K_fact.LDL.Lx) + length(pad.pdat.K_fact.LDL.d)
    end

    elapsed_time = time() - sc.start_time

    stats = GenericExecutionStats(
      QM,
      status = status,
      solution = x,
      objective = itd.minimize ? itd.pri_obj : -itd.pri_obj,
      dual_feas = res.rcNorm,
      primal_feas = res.rbNorm,
      multipliers = multipliers,
      multipliers_L = multipliers_L,
      multipliers_U = multipliers_U,
      iter = cnts.k,
      elapsed_time = elapsed_time,
      solver_specific = solver_specific,
    )
  end
  return stats
end

function ripqp(
  LLS::LLSModel{T0};
  mode::Symbol = :mono,
  sp::SolverParams = (mode == :mono) ? K2LDLParams{T0}() : K2LDLParams{Float32}(),
  kwargs...,
) where {T0 <: Real}
  sp.δ0 = 0.0 # equality constraints of least squares as QPs are already regularized
  FLLS = FeasibilityFormNLS(LLS)
  stats =
    ripqp(QuadraticModel(FLLS, FLLS.meta.x0, name = LLS.meta.name); mode = mode, sp = sp, kwargs...)
  n = LLS.meta.nvar
  x, r = stats.solution[1:n], stats.solution[(n + 1):end]
  solver_sp = stats.solver_specific
  solver_sp[:r] = r
  return GenericExecutionStats(
    LLS,
    status = stats.status,
    solution = x,
    objective = stats.objective,
    dual_feas = stats.dual_feas,
    primal_feas = stats.primal_feas,
    multipliers = stats.multipliers,
    multipliers_L = stats.multipliers_L,
    multipliers_U = stats.multipliers_U,
    iter = stats.iter,
    elapsed_time = stats.elapsed_time,
    solver_specific = solver_sp,
  )
end

end
