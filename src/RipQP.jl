module RipQP

using DelimitedFiles, LinearAlgebra, MatrixMarket, Quadmath, SparseArrays, TimerOutputs

using HSL,
  Krylov,
  LDLFactorizations,
  LimitedLDLFactorizations,
  LinearOperators,
  LLSModels,
  NLPModelsModifiers,
  OperatorScaling,
  QuadraticModels,
  SolverCore,
  SparseMatricesCOO

using Requires
function __init__()
  @require CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba" include("gpu_utils.jl")
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

keep_init_prec(mode::Symbol) = (mode == :mono || mode == :ref || mode == :zoom)

include("mono_solver.jl")
include("two_steps_solver.jl")
include("three_steps_solver.jl")

function ripqp(
  QM::QuadraticModel{T, S},
  ap::AbstractRipQPParameters{T};
  ps::Bool = true,
  scaling::Bool = true,
  history::Bool = false,
  display::Bool = true,
  kwargs...,
) where {T, S <: AbstractVector{T}}
  start_time = time()
  # conversion function if QM.data.H and QM.data.A are not in the type required by sp
  QMi, ps, scaling = convert_QM(QM, ps, scaling, ap.sp, display)
  if ps
    stats_ps = presolve(QMi, fixed_vars_only = !ps)
    stats_ps.status != :unknown && (return stats_ps)
    QMps = stats_ps.solver_specific[:presolvedQM]
  else
    QMps = QMi
  end

  QM_inner, stats_inner, idi = get_inner_model_data(QM, QMps, ps, scaling, history)

  solver =
    RipQPSolver(QM_inner, ap; history = history, scaling = scaling, display = display, kwargs...)

  SolverCore.solve!(solver, QM_inner, stats_inner, ap)

  return get_stats_outer(stats_inner, QM, QMps, solver.id, idi, start_time, ps)
end

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
- `scaling :: Bool`: activate/deactivate scaling of A and Q in `QM`
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
    Following algorithm.
    `solve_method2 :: Union{Nothing, SolveMethod}` and  `solve_method3 :: Union{Nothing, SolveMethod}`
    should be used with `sp2` and `sp3` to choose their respective solve method.
    If they are `nothing`, then the solve method used is `solve_method`. 
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
  QM::QuadraticModel{T, S};
  mode::Symbol = :mono,
  sp::SolverParams = keep_init_prec(mode) ? K2LDLParams{T}() : K2LDLParams{Float32}(),
  sp2::Union{Nothing, SolverParams} = nothing,
  sp3::Union{Nothing, SolverParams} = nothing,
  solve_method::SolveMethod = PC(),
  solve_method2::Union{Nothing, SolveMethod} = nothing,
  solve_method3::Union{Nothing, SolveMethod} = nothing,
  kwargs...,
) where {T, S <: AbstractVector{T}}
  if mode == :mono
    ap = RipQPMonoParameters(sp, solve_method)
  else
    T1 = solver_type(sp)
    T2 = isnothing(sp2) ? next_type(T1, T) : solver_type(sp2)
    isnothing(sp2) && (sp2 = eval(typeof(sp).name.name){T2}())
    isnothing(solve_method2) && (solve_method2 = solve_method)
    if T2 != T || !isnothing(sp3)
      isnothing(sp3) && (sp3 = eval(typeof(sp2).name.name){T}())
      isnothing(solve_method3) && (solve_method3 = solve_method2)
      ap = RipQPTripleParameters(sp, sp2, sp3, solve_method, solve_method2, solve_method3)
    else
      ap = RipQPDoubleParameters(sp, sp2, solve_method, solve_method2)
    end
  end
  return ripqp(QM, ap; mode = mode, kwargs...)
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
