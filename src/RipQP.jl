module RipQP

using DelimitedFiles, LinearAlgebra, MatrixMarket, Quadmath, SparseArrays, Statistics, TimerOutputs

using Krylov,
  LDLFactorizations,
  LinearOperators,
  LLSModels,
  NLPModelsModifiers,
  QuadraticModels,
  SolverCore,
  SparseMatricesCOO

using Requires
function __init__()
  @require CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba" include("gpu_utils.jl")
end

export ripqp, RipQPSolver, default_parameters

include("types_definition.jl")
include("iterations/iterations.jl")
include("refinement.jl")
include("data_initialization.jl")
include("starting_points.jl")
include("scaling.jl")
include("multi_precision.jl")
include("utils.jl")

const to = TimerOutput()

function default_parameters(T::DataType)
  return (
    r=Float64(0.999),
    γ=Float64(0.05),
    kc=0
  )
end 

"""
    stats = ripqp(QM :: QuadraticModel{T0};
                  itol = InputTol(T0), scaling = true, ps = true,
                  normalize_rtol = true, kc = 0, mode = :mono, perturb = false,
                  Timulti = Float32, 
                  sp = (mode == :mono) ? K2LDLParams{T0}() : K2LDLParams{Timulti}(),
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
- `Timulti :: DataType`: initial floating-point format to solve the QP (only usefull in multi-precision),
    it should be lower than the QP precision
- `sp :: SolverParams` : choose a solver to solve linear systems that occurs at each iteration and during the 
    initialization, see [`RipQP.SolverParams`](@ref)
- `sp2 :: Union{Nothing, SolverParams}` and `sp3 :: Union{Nothing, SolverParams}` : choose second and third solvers
    to solve linear systems that occurs at each iteration in the second and third solving phase when `mode != :mono`, 
    leave to `nothing` if you want to keep using `sp`. 
- `solve_method :: SolveMethod` : method used to solve the system at each iteration, use `solve_method = PC()` to 
    use the Predictor-Corrector algorithm (default), and use `solve_method = IPF()` to use the Infeasible Path 
    Following algorithm
- `history :: Bool` : set to true to return the primal and dual norm histories, the primal-dual relative difference
    history, and the number of products if using a Krylov method in the `solver_specific` field of the 
    [GenericExecutionStats](https://juliasmoothoptimizers.github.io/SolverCore.jl/dev/reference/#SolverCore.GenericExecutionStats)
- `w :: SystemWrite`: configure writing of the systems to solve (no writing is done by default), see [`RipQP.SystemWrite`](@ref)
- `display::Bool`: activate/deactivate iteration data display

You can also use `ripqp` to solve a [LLSModel](https://juliasmoothoptimizers.github.io/LLSModels.jl/stable/#LLSModels.LLSModel):

    stats = ripqp(LLS::LLSModel{T0}; mode = :mono, Timulti = Float32,
                  sp = (mode == :mono) ? K2LDLParams{T0}() : K2LDLParams{Timulti}(), 
                  kwargs...) where {T0 <: Real}
"""
function ripqp(QM0::QuadraticModel{T0}, params::NamedTuple;kwargs...) where {T0 <: Real}
  params = merge(params, default_parameters(T0))
  return ripqp(QM0;solve_method=IPF(r=params.r, γ=params.γ), kc=params.kc, kwargs...)
end

function ripqp(
  QM0::QuadraticModel{T0};
  itol::InputTol{T0, Int} = InputTol(T0),
  scaling::Bool = true,
  ps::Bool = true,
  normalize_rtol::Bool = true,
  kc::I = 0,
  perturb::Bool = false,
  mode::Symbol = :mono,
  Timulti::DataType = Float32,
  sp::SolverParams = (mode == :mono) ? K2LDLParams{T0}() : K2LDLParams{Timulti}(),
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
    typeof(solve_method) <: IPF &&
      kc != 0 &&
      error("IPF method should not be used with centrality corrections")
    iconf = InputConfig(
      mode,
      Timulti,
      scaling,
      ps,
      normalize_rtol,
      kc,
      perturb,
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
      stats_ps = presolve(QM0)
      if stats_ps.status == :unknown
        QM = stats_ps.solver_specific[:presolvedQM]
      else
        return stats_ps
      end
    else
      QM = QM0
    end

    # allocate workspace
    sc, idi, fd_T0, id, ϵ, res, itd, dda, pt, sd, spd, cnts, T =
      @timeit_debug to "allocate workspace" allocate_workspace(QM, iconf, itol, start_time, T0)

    if iconf.scaling
      scaling!(fd_T0, id, sd, T0(1.0e-5))
    end

    # extra workspace for multi mode
    if iconf.mode == :multi || iconf.mode == :multizoom || iconf.mode == :multiref
      if iconf.Timulti == Float32
        fd32, ϵ32 = allocate_extra_workspace_32(itol, iconf, fd_T0)
      end
      if T0 == Float128
        fd64, ϵ64 = allocate_extra_workspace_64(itol, iconf, fd_T0)
      end
    end

    # initialize
    if iconf.mode == :multi || iconf.mode == :multizoom || iconf.mode == :multiref
      if iconf.Timulti == Float32
        pad = initialize!(fd32, id, res, itd, dda, pt, spd, ϵ32, sc, iconf, cnts, T0)
        if T0 == Float128
          set_tol_residuals!(ϵ64, Float64(res.rbNorm), Float64(res.rcNorm))
          T = Float32
        end
      elseif iconf.Timulti == Float64
        pad = initialize!(fd64, id, res, itd, dda, pt, spd, ϵ64, sc, iconf, cnts, T0)
      else
        error("not a valid Timulti")
      end
      set_tol_residuals!(ϵ, T0(res.rbNorm), T0(res.rcNorm))
    elseif iconf.mode == :mono || iconf.mode == :zoom || iconf.mode == :ref
      pad = initialize!(fd_T0, id, res, itd, dda, pt, spd, ϵ, sc, iconf, cnts, T0)
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

    if (iconf.mode == :multi || iconf.mode == :multizoom || iconf.mode == :multiref)
      if iconf.Timulti == Float32
        # iter in Float32 then convert data to Float64
        pt, itd, res, dda, pad = iter_and_update_T!(
          iconf.sp,
          iconf.sp2,
          pt,
          itd,
          fd32,
          (T0 == Float64) ? fd_T0 : fd64,
          id,
          res,
          sc,
          dda,
          pad,
          ϵ32,
          ϵ,
          cnts,
          itol.max_iter32,
          display,
        )
      end
      if T0 == Float128
        # iters in Float64 then convert data to Float128
        pt, itd, res, dda, pad = iter_and_update_T!(
          (iconf.sp2 === nothing) ? iconf.sp : iconf.sp2,
          iconf.sp3,
          pt,
          itd,
          fd64,
          fd_T0,
          id,
          res,
          sc,
          dda,
          pad,
          ϵ64,
          ϵ,
          cnts,
          itol.max_iter64,
          display,
        )
      end
      sc.max_iter = itol.max_iter
    end

    ## iter T0
    # refinement
    if !sc.optimal
      if iconf.mode == :zoom || iconf.mode == :ref
        ϵz = Tolerances(
          T(1),
          T(itol.ϵ_rbz),
          T(itol.ϵ_rbz),
          T(ϵ.tol_rb * T(itol.ϵ_rbz / itol.ϵ_rb)),
          one(T),
          T(itol.ϵ_μ),
          T(itol.ϵ_Δx),
          iconf.normalize_rtol,
        )
        iter!(pt, itd, fd_T0, id, res, sc, dda, pad, ϵz, cnts, T0, display)
        sc.optimal = false

        fd_ref, pt_ref =
          fd_refinement(fd_T0, id, res, itd.Δxy, pt, itd, ϵ, dda, pad, spd, cnts, T0, iconf.mode)
        iter!(pt_ref, itd, fd_ref, id, res, sc, dda, pad, ϵ, cnts, T0, display)
        update_pt_ref!(fd_ref.Δref, pt, pt_ref, res, id, fd_T0, itd)

      elseif iconf.mode == :multizoom || iconf.mode == :multiref
        spd = convert(StartingPointData{T0, typeof(pt.x)}, spd)
        fd_ref, pt_ref = fd_refinement(
          fd_T0,
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
        iter!(pt_ref, itd, fd_ref, id, res, sc, dda, pad, ϵ, cnts, T0, display)
        update_pt_ref!(fd_ref.Δref, pt, pt_ref, res, id, fd_T0, itd)

      elseif iconf.mode == :mono || iconf.mode == :multi
        # iters T0, no refinement
        iter!(pt, itd, fd_T0, id, res, sc, dda, pad, ϵ, cnts, T0, display)
      end
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

    if iconf.presolve
      x = similar(QM0.meta.x0)
      postsolve!(QM0, QM, pt.x, x)
      nrm = length(QM.xrm)
    else
      x = pt.x[1:(idi.nvar)]
      nrm = 0
    end

    multipliers, multipliers_L, multipliers_U =
      get_multipliers(pt.s_l, pt.s_u, id.ilow, id.iupp, id.nvar, pt.y, idi, nrm)

    if typeof(res) <: ResidualsHistory
      solver_specific = Dict(
        :absolute_iter_cnt => cnts.k,
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
      )
    else
      solver_specific = Dict(:absolute_iter_cnt => cnts.k, :pdd => itd.pdd, :nvar_slack => id.nvar)
    end

    if typeof(pad) <: PreallocatedDataK2Krylov && typeof(pad.pdat) <: LDLData
      solver_specific[:nnzLDL] = length(pad.pdat.K_fact.Lx) + length(pad.pdat.K_fact.d)
    end

    elapsed_time = time() - sc.start_time

    stats = GenericExecutionStats(
      status,
      QM,
      solution = x,
      objective = itd.minimize ? itd.pri_obj : -itd.pri_obj,
      dual_feas = res.rcNorm,
      primal_feas = res.rbNorm,
      multipliers = multipliers,
      multipliers_L = multipliers_L,
      multipliers_U = multipliers_U,
      iter = cnts.km,
      elapsed_time = elapsed_time,
      solver_specific = solver_specific,
    )
  end
  return stats
end

function ripqp(
  LLS::LLSModel{T0};
  mode::Symbol = :mono,
  Timulti::DataType = Float32,
  sp::SolverParams = (mode == :mono) ? K2LDLParams{T0}() : K2LDLParams{Timulti}(),
  kwargs...,
) where {T0 <: Real}
  sp.δ0 = 0.0 # equality constraints of least squares as QPs are already regularized
  FLLS = FeasibilityFormNLS(LLS)
  stats = ripqp(
    QuadraticModel(FLLS, FLLS.meta.x0, name = LLS.meta.name);
    mode = mode,
    Timulti = Timulti,
    sp = sp,
    kwargs...,
  )
  n = LLS.meta.nvar
  x, r = stats.solution[1:n], stats.solution[(n + 1):end]
  solver_sp = stats.solver_specific
  solver_sp[:r] = r
  return GenericExecutionStats(
    stats.status,
    LLS,
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
