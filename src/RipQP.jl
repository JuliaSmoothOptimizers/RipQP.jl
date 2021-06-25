module RipQP

using Base: Float64
using DelimitedFiles, LinearAlgebra, MatrixMarket, Quadmath, SparseArrays, Statistics

using LDLFactorizations, LLSModels, NLPModelsModifiers, QuadraticModels, SolverCore

export ripqp

include("types_definition.jl")
include("iterations/iterations.jl")
include("refinement.jl")
include("data_initialization.jl")
include("starting_points.jl")
include("scaling.jl")
include("multi_precision.jl")
include("utils.jl")

"""
    stats = ripqp(QM :: QuadraticModel; iconf :: InputConfig{Int} = InputConfig(),
                  itol :: InputTol{Tu, Int} = InputTol(),
                  display :: Bool = true) where {Tu<:Real}

Minimize a convex quadratic problem. Algorithm stops when the criteria in pdd, rb, and rc are valid.
Returns a [GenericExecutionStats](https://juliasmoothoptimizers.github.io/SolverCore.jl/dev/reference/#SolverCore.GenericExecutionStats) 
containing information about the solved problem.

- `QM :: QuadraticModel`: problem to solve
- `iconf :: InputConfig{Int}`: input RipQP configuration. See [`RipQP.InputConfig`](@ref).
- `itol :: InputTol{T, Int}` input Tolerances for the stopping criteria. See [`RipQP.InputTol`](@ref).
- `display::Bool`: activate/deactivate iteration data display

You can also use `ripqp` to solve a [LLSModel](https://juliasmoothoptimizers.github.io/LLSModels.jl/stable/#LLSModels.LLSModel):

    stats = ripqp(LLS :: LLSModel; iconf :: InputConfig{Int} = InputConfig(),
                  itol :: InputTol{Tu, Int} = InputTol(),
                  display :: Bool = true) where {Tu<:Real}
"""
function ripqp(
  QM::QuadraticModel;
  iconf::InputConfig{Int} = InputConfig(),
  itol::InputTol{Tu, Int} = InputTol(),
  display::Bool = true,
) where {Tu <: Real}

  start_time = time()
  elapsed_time = 0.0
  T0 = eltype(QM.data.c)

  # allocate workspace
  if iconf.mode == :mono
    sc, idi, fd_T0, id, ϵ, res, sd, T = allocate_workspace(QM, iconf, itol, start_time, T0)
  elseif iconf.mode == :multi && T0 == Float64
    sc, idi, fd_T0, id, ϵ, res, sd, T, ϵ32, fd32 = allocate_workspace(QM, iconf, itol, start_time, T0)
  elseif iconf.mode == :multi && T0 == Float128
    sc, idi, fd_T0, id, ϵ, res, sd, T, ϵ32, fd32, ϵ64, fd64 = allocate_workspace(QM, iconf, itol, start_time, T0)
  end

  # initialize
  if iconf.mode == :multi
    itd, ϵ32, dda, pad, pt, sc, cnts = init_params(fd32, id, res, ϵ32, sc, iconf, T0)
    set_tol_residuals!(ϵ, T0(res.rbNorm), T0(res.rcNorm))
    if T0 == Float128
      set_tol_residuals!(ϵ64, Float64(res.rbNorm), Float64(res.rcNorm))
      T = Float32
    end
  elseif iconf.mode == :mono
    itd, ϵ, dda, pad, pt, sc, cnts = init_params(fd_T0, id, res, ϵ, sc, iconf, T0)
  end

  Δt = time() - start_time
  sc.tired = Δt > itol.max_time

  # display
  if display == true
    @info log_header(
      [:k, :pri_obj, :pdd, :rbNorm, :rcNorm, :α_pri, :α_du, :μ],
      [Int, T, T, T, T, T, T, T, T, T, T, T],
      hdr_override = Dict(
        :k => "iter",
        :pri_obj => "obj",
        :pdd => "rgap",
        :rbNorm => "‖rb‖",
        :rcNorm => "‖rc‖",
      ),
    )
    @info log_row(
      Any[cnts.k, itd.pri_obj, itd.pdd, res.rbNorm, res.rcNorm, zero(T), zero(T), itd.μ],
    )
  end

  if iconf.mode == :multi
    # iter in Float32 then convert data to Float64
    pt, itd, res, dda, pad = iter_and_update_T!(
      pt,
      itd,
      fd32,
      id,
      res,
      sc,
      dda,
      pad,
      ϵ32,
      ϵ,
      cnts,
      itol.max_iter32,
      Float64,
      display,
    )

    if T0 == Float128
      # iters in Float64 then convert data to Float128
      pt, itd, res, dda, pad = iter_and_update_T!(
        pt,
        itd,
        fd64,
        id,
        res,
        sc,
        dda,
        pad,
        ϵ64,
        ϵ,
        cnts,
        itol.max_iter64,
        Float128,
        display,
      )
    end
    sc.max_iter = itol.max_iter
  end

  ## iter T0
  # refinement
  if !sc.optimal
    if iconf.refinement == :zoom || iconf.refinement == :ref
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
        fd_refinement(fd_T0, id, res, itd.Δxy, pt, itd, ϵ, dda, pad, cnts, T0, iconf.refinement)
      iter!(pt_ref, itd, fd_ref, id, res, sc, dda, pad, ϵ, cnts, T0, display)
      update_pt_ref!(fd_ref.Δref, pt, pt_ref, res, id, fd_T0, itd)

    elseif iconf.refinement == :multizoom || iconf.refinement == :multiref
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
        cnts,
        T0,
        iconf.refinement,
        centering = true,
      )
      iter!(pt_ref, itd, fd_ref, id, res, sc, dda, pad, ϵ, cnts, T0, display)
      update_pt_ref!(fd_ref.Δref, pt, pt_ref, res, id, fd_T0, itd)

    else
      # iters T0, no refinement
      iter!(pt, itd, fd_T0, id, res, sc, dda, pad, ϵ, cnts, T0, display)
    end
  end

  if iconf.scaling
    pt, pri_obj, res = post_scale(
      sd.d1,
      sd.d2,
      sd.d3,
      pt,
      res,
      fd_T0,
      id,
      itd.Qx,
      itd.ATy,
      itd.Ax,
      itd.cTx,
      itd.pri_obj,
      itd.dual_obj,
      itd.xTQx_2,
    )
  end

  if cnts.k >= itol.max_iter
    status = :max_iter
  elseif sc.tired
    status = :max_time
  elseif sc.optimal
    status = :acceptable
  else
    status = :unknown
  end
  multipliers, multipliers_L, multipliers_U =
    get_multipliers(pt.s_l, pt.s_u, id.ilow, id.iupp, id.nvar, pt.y, idi)

  elapsed_time = time() - sc.start_time

  stats = GenericExecutionStats(
    status,
    QM,
    solution = pt.x[1:(idi.nvar)],
    objective = itd.pri_obj,
    dual_feas = res.rcNorm,
    primal_feas = res.rbNorm,
    multipliers = multipliers,
    multipliers_L = multipliers_L,
    multipliers_U = multipliers_U,
    iter = cnts.km,
    elapsed_time = elapsed_time,
    solver_specific = Dict(:absolute_iter_cnt => cnts.k),
  )
  return stats
end

function ripqp(LLS::LLSModel; kwargs...)
  FLLS = FeasibilityFormNLS(LLS)
  return ripqp(QuadraticModel(FLLS, FLLS.meta.x0, name = LLS.meta.name); kwargs...)
end

end
