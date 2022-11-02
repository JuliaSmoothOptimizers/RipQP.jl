module RipQP

using DelimitedFiles, LinearAlgebra, MatrixMarket, Quadmath, SparseArrays, TimerOutputs

using HSL,
  Krylov,
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

function RipQPSolver(
  QM::AbstractQuadraticModel{T0, S0};
  itol::InputTol{T0, Int} = InputTol(T0),
  scaling::Bool = true,
  ps::Bool = true,
  normalize_rtol::Bool = true,
  kc::I = 0,
  perturb::Bool = false,
  mode::Symbol = :mono,
  early_multi_stop::Bool = true,
  sp::SolverParams = keep_init_prec(mode) ? K2LDLParams{T0}() : K2LDLParams{Float32}(),
  sp2::Union{Nothing, SolverParams} = nothing,
  sp3::Union{Nothing, SolverParams} = nothing,
  solve_method::SolveMethod = PC(),
  solve_method2::Union{Nothing, SolveMethod} = nothing,
  solve_method3::Union{Nothing, SolveMethod} = nothing,
  history::Bool = false,
  w::SystemWrite = SystemWrite(),
  display::Bool = true,
) where {T0 <: Real, S0 <: AbstractVector{T0}, I <: Integer}

  start_time = time()
  elapsed_time = 0.0
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
    ps || (QM.data.H isa SparseMatrixCOO && QM.data.A isa SparseMatrixCOO),
    normalize_rtol,
    kc,
    perturb,
    QM.meta.minimize,
    history,
    w,
  )

  # allocate workspace
  sc, fd_T0, id, ϵ_T0, res, itd, pt, sd, spd, cnts, T =
    allocate_workspace(QM, iconf, itol, start_time, T0, sp)

  if iconf.scaling
    scaling!(fd_T0, id, sd, T0(1.0e-5))
  end

  # extra workspace for multi mode
  if (iconf.mode == :multi || iconf.mode == :multizoom || iconf.mode == :multiref)
    T2 = isnothing(sp2) ? next_type(Ti, T0) : solver_type(sp2) # eltype of sp2
    # if the 2nd solver is nothing:
    isnothing(sp2) && (sp2 = eval(typeof(sp).name.name){T2}())
    isnothing(solve_method2) && (solve_method2 = solve_method)
    fd1, ϵ1 = allocate_extra_workspace1(Ti, itol, iconf, fd_T0)
    if T2 != T0 || !isnothing(sp3)
      fd2, ϵ2 = allocate_extra_workspace2(T2, itol, iconf, fd_T0)
      # if the 3nd solver is nothing:
      isnothing(sp3) && (sp3 = eval(typeof(sp2).name.name){T0}())
      isnothing(solve_method3) && (solve_method3 = solve_method2)
    end
  elseif iconf.mode == :ref || iconf.mode == :zoom
    isnothing(sp2) && (sp2 = eval(typeof(sp).name.name){T}())
    isnothing(solve_method2) && (solve_method2 = solve_method)
    fd1 = fd_T0
    ϵ1 = Tolerances(
      T(1),
      T(itol.ϵ_rbz),
      T(itol.ϵ_rbz),
      T(ϵ_T0.tol_rb * T(itol.ϵ_rbz / itol.ϵ_rb)),
      one(T),
      T(itol.ϵ_μ),
      T(itol.ϵ_Δx),
      iconf.normalize_rtol,
    )
  end

  if isnothing(sp2)
    fd1, ϵ1 = nothing, nothing
    fd = fd_T0
  else
    fd = fd1
  end
  if isnothing(sp3)
    fd2, ϵ2 = nothing, nothing
  end
  
  S = change_vector_eltype(S0, T)
  dda = DescentDirectionAllocs(id, solve_method, S)

  pad = PreallocatedData(sp, fd, id, itd, pt, iconf)

  pfd = PreallocatedFloatData(pt, res, itd, dda, pad)

  cnts.time_allocs = time() - start_time

  if isnothing(sp2)
    solver = RipQPSolver(
      QM,
      id,
      iconf,
      itol,
      sd,
      spd,
      sc,
      cnts,
      display,
      fd_T0,
      ϵ_T0,
      pfd,
    )
  else
    if isnothing(sp3)
      solver = RipQPDoubleSolver(
        QM,
        id,
        iconf,
        itol,
        sd,
        spd,
        sc,
        cnts,
        display,
        fd1,
        ϵ1,
        sp,
        solve_method,  
        fd_T0,
        ϵ_T0,
        sp2,
        solve_method2,
        pfd,
      )
    else
      solver = RipQPTripleSolver(
        QM,
        id,
        iconf,
        itol,
        sd,
        spd,
        sc,
        cnts,
        display,
        fd1,
        ϵ1,
        sp,
        solve_method,
        fd2,
        ϵ2,
        sp2,
        solve_method2,
        fd_T0,
        ϵ_T0,
        sp3,
        solve_method3,
        pfd,
      )
    end
  end
  return solver
end

# function SolverCore.solve!(
#   solver::RipQPSolver{T0},
#   QM::AbstractQuadraticModel{T0},
#   stats::GenericExecutionStats{T0};
# ) where {T0}
#   start_time = time()

#   id = solver.id
#   iconf, itol = solver.iconf, solver.itol
#   sd, spd = solver.sd, solver.spd
#   sc, cnts = solver.sc, solver.cnts
#   display = solver.display
#   # sp, sp2, sp3 = solver.sp, solver.sp2, solver.sp3
#   sp = solver.sp
#   # solve_method, solve_method2, solve_method3 = solver.solve_method, solver.solve_method2, solver.solve_method3
#   solve_method = solver.solve_method
#   # fd_T0, ϵ_T0 = solver.fd_T0, solver.ϵ_T0
#   pfd = solver.pfd
#   pt, res, itd, dda, pad = pfd.pt, pfd.res, pfd.itd, pfd.dda, pfd.pad 
#   mode = iconf.mode
#   # if !isnothing(sp2)
#   #   fd1, ϵ1 = solver.fd1, solver.ϵ1
#   #   fd, ϵ = fd1, ϵ1
#   #   sc.max_iter = itol.max_iter1 # set max_iter for 1st solver
#   #   cnts.last_sp = false # sp is not the last sp because sp2 is not nothing
#   # else
#   #   fd, ϵ = fd_T0, ϵ_T0
#   # end
#   # if !isnothing(sp3)
#   #   fd2, ϵ2 = solver.fd2, solver.ϵ2
#   # end
#   fd, ϵ = solver.fd, solver.ϵ
#   fd_T0, ϵ_T0 = fd, ϵ

#   # initialize data (some allocations for the pad creation)
#   pad = initialize!(fd, id, res, itd, dda, pad, pt, spd, ϵ, sc, iconf, cnts, T0)
#   # if (iconf.mode == :multi || iconf.mode == :multizoom || iconf.mode == :multiref)
#   #   # set intermediate tolerances
#   #   set_tol_residuals!(ϵ_T0, T0(res.rbNorm), T0(res.rcNorm))
#   #   T2 = solver_type(sp2)
#   #   !isnothing(sp3) && set_tol_residuals!(ϵ2, T2(res.rbNorm), T2(res.rcNorm))
#   # end

#   Δt = time() - start_time
#   sc.tired = Δt > itol.max_time

#   # display
#   if display == true
#     # @timeit_debug to "display" begin
#       T = solver_type(sp)
#       show_used_solver(pad)
#       setup_log_header(pad)
#       show_log_row(pad, itd, res, cnts, zero(T), zero(T))
#     # end
#   end

#   last_iter = mode == :mono # do not stop early in mono mode

#   # IPM iterations 1st solver (mono mode: only this call to the iter! function)
#   iter!(pt, itd, fd, id, res, sc, dda, pad, ϵ, cnts, iconf, display, last_iter = last_iter)

#   # initialize iteration counters
#   iters_sp, iters_sp2, iters_sp3 = cnts.k, 0, 0

#   # if !isnothing(sp2) # setup data for 2nd solver
#   #   if isnothing(sp3)
#   #     fd = fd_T0
#   #     ϵ = ϵ_T0
#   #     sc.max_iter = itol.max_iter # set max_iter for 2nd solver
#   #     cnts.last_sp = true # sp2 is the last sp because sp3 is nothing
#   #   else
#   #     fd = fd2
#   #     ϵ = ϵ2
#   #     sc.max_iter = itol.max_iter2 # set max_iter for 2nd solver
#   #     cnts.last_sp = false # sp2 is not the last sp because sp3 is not nothing
#   #   end
#   #   pt, itd, res, dda, pad =
#   #     convert_types(T2, pt, itd, res, dda, pad, sp, sp2, id, fd, solve_method, solve_method2)
#   #   sc.optimal = itd.pdd < ϵ_T0.pdd && res.rbNorm < ϵ_T0.tol_rb && res.rcNorm < ϵ_T0.tol_rc
#   #   sc.small_μ = itd.μ < ϵ_T0.μ
#   #   display && show_used_solver(pad)
#   #   display && setup_log_header(pad)
#   # end

#   # if !isnothing(sp3) # iter! with 2nd solver, setup data 3rd solver
#   #   # IPM iterations 2nd solver starting from pt
#   #   iter!(
#   #     pt,
#   #     itd,
#   #     fd,
#   #     id,
#   #     res,
#   #     sc,
#   #     dda,
#   #     pad,
#   #     ϵ,
#   #     cnts,
#   #     iconf,
#   #     display,
#   #     last_iter = !isnothing(sp3),
#   #   )

#   #   # iteration counter for 2nd solver
#   #   iters_sp2 = cnts.k - iters_sp

#   #   fd = fd_T0
#   #   ϵ = ϵ_T0
#   #   pt, itd, res, dda, pad =
#   #     convert_types(T0, pt, itd, res, dda, pad, sp2, sp3, id, fd, solve_method2, solve_method3)
#   #   sc.optimal = itd.pdd < ϵ_T0.pdd && res.rbNorm < ϵ_T0.tol_rb && res.rcNorm < ϵ_T0.tol_rc
#   #   sc.small_μ = itd.μ < ϵ_T0.μ
#   #   display && show_used_solver(pad)
#   #   display && setup_log_header(pad)

#   #   # set max_iter for 3rd solver
#   #   sc.max_iter = itol.max_iter
#   #   cnts.last_sp = true # sp3 is the last sp
#   # end

#   # IPM iterations 3rd solver: different following the use of mode multi, multizoom, multiref
#   # starting from pt
#   # if !sc.optimal && mode == :multi
#   #   iter!(pt, itd, fd, id, res, sc, dda, pad, ϵ, cnts, iconf, display)
#   # elseif !sc.optimal && (mode == :multizoom || mode == :multiref)
#   #   spd = convert(StartingPointData{T0, typeof(pt.x)}, spd)
#   #   fd_ref, pt_ref = fd_refinement(
#   #     fd,
#   #     id,
#   #     res,
#   #     itd.Δxy,
#   #     pt,
#   #     itd,
#   #     ϵ,
#   #     dda,
#   #     pad,
#   #     spd,
#   #     cnts,
#   #     iconf.mode,
#   #     centering = true,
#   #   )
#   #   iter!(pt_ref, itd, fd_ref, id, res, sc, dda, pad, ϵ, cnts, iconf, display)
#   #   update_pt_ref!(fd_ref.Δref, pt, pt_ref, res, id, fd_T0, itd)
#   # elseif iconf.mode == :zoom || iconf.mode == :ref
#   #   ϵ = ϵ_T0
#   #   sc.optimal = false
#   #   fd_ref, pt_ref =
#   #     fd_refinement(fd, id, res, itd.Δxy, pt, itd, ϵ, dda, pad, spd, cnts, iconf.mode)
#   #   iter!(pt_ref, itd, fd_ref, id, res, sc, dda, pad, ϵ, cnts, iconf, display)
#   #   update_pt_ref!(fd_ref.Δref, pt, pt_ref, res, id, fd_T0, itd)
#   # end

  
#   # update number of iterations for each solver 
#   if isnothing(sp3) && !isnothing(sp2)
#     iters_sp2 = cnts.k - iters_sp
#   elseif !isnothing(sp3)
#     iters_sp3 = cnts.k - iters_sp2
#   end

#   if iconf.scaling
#     post_scale!(sd, pt, res, fd_T0, id, itd)
#   end
#   set_ripqp_stats!(stats, pt, res, pad, itd, id, sc, cnts, itol.max_iter, sp2, sp3)
#   return stats
# end

function SolverCore.solve!(
  solver::RipQPTripleSolver{T},
  QM::AbstractQuadraticModel{T},
  stats::GenericExecutionStats{T};
) where {T}
  id = solver.id
  iconf, itol = solver.iconf, solver.itol
  sd, spd = solver.sd, solver.spd
  sc, cnts = solver.sc, solver.cnts
  display = solver.display
  sp, sp2, sp3 = solver.sp, solver.sp2, solver.sp3
  solve_method, solve_method2, solve_method3 = solver.solve_method, solver.solve_method2, solver.solve_method3
  pfd = solver.pfd
  pt, res, itd, dda, pad = pfd.pt, pfd.res, pfd.itd, pfd.dda, pfd.pad 
  fd1, ϵ1 = solver.fd1, solver.ϵ1
  fd2, ϵ2 = solver.fd2, solver.ϵ2
  fd3, ϵ3 = solver.fd3, solver.ϵ3
  cnts.time_solve = time()
  sc.max_iter = itol.max_iter1 # set max_iter for 1st solver
  cnts.last_sp = false # sp is not the last sp because sp2 is not nothing
  mode = iconf.mode

  T2, T3 = solver_type(sp2), solver_type(sp3)
  @assert T3 == T
  # initialize data
  initialize!(fd1, id, res, itd, dda, pad, pt, spd, ϵ1, sc, cnts, itol.max_time, display, T)
  if (iconf.mode == :multi || iconf.mode == :multizoom || iconf.mode == :multiref)
    set_tol_residuals!(ϵ2, T2(res.rbNorm), T2(res.rcNorm))
    set_tol_residuals!(ϵ3, T3(res.rbNorm), T3(res.rcNorm))
  end

  # IPM iterations 1st solver (mono mode: only this call to the iter! function)
  iter!(pt, itd, fd1, id, res, sc, dda, pad, ϵ1, cnts, iconf, display, last_iter = false)
  cnts.iters_sp = cnts.k

  sc.max_iter = itol.max_iter # set max_iter for 2nd solver
  cnts.last_sp = true # sp2 is the last sp because sp3 is nothing
  pt, itd, res, dda, pad =
    convert_types(T2, pt, itd, res, dda, pad, sp, sp2, id, fd2, solve_method, solve_method2)
  sc.optimal = itd.pdd < ϵ2.pdd && res.rbNorm < ϵ2.tol_rb && res.rcNorm < ϵ2.tol_rc
  sc.small_μ = itd.μ < ϵ2.μ
  display && show_used_solver(pad)
  display && setup_log_header(pad)

  iter!(pt, itd, fd2, id, res, sc, dda, pad, ϵ2, cnts, iconf, display, last_iter = false)

  cnts.iters_sp2 = cnts.k - cnts.iters_sp
  pt, itd, res, dda, pad =
    convert_types(T3, pt, itd, res, dda, pad, sp2, sp3, id, fd3, solve_method2, solve_method3)
  sc.optimal = itd.pdd < ϵ3.pdd && res.rbNorm < ϵ3.tol_rb && res.rcNorm < ϵ3.tol_rc
  sc.small_μ = itd.μ < ϵ3.μ
  display && show_used_solver(pad)
  display && setup_log_header(pad)

  # set max_iter for 3rd solver
  sc.max_iter = itol.max_iter
  cnts.last_sp = true # sp3 is the last sp

  if !sc.optimal && mode == :multi
    iter!(pt, itd, fd3, id, res, sc, dda, pad, ϵ3, cnts, iconf, display)
  elseif !sc.optimal && (mode == :multizoom || mode == :multiref)
    spd = convert(StartingPointData{T, typeof(pt.x)}, spd)
    fd_ref, pt_ref = fd_refinement(
      fd3,
      id,
      res,
      itd.Δxy,
      pt,
      itd,
      ϵ3,
      dda,
      pad,
      spd,
      cnts,
      iconf.mode,
      centering = true,
    )
    iter!(pt_ref, itd, fd_ref, id, res, sc, dda, pad, ϵ3, cnts, iconf, display)
    update_pt_ref!(fd_ref.Δref, pt, pt_ref, res, id, fd3, itd)
  elseif iconf.mode == :zoom || iconf.mode == :ref
    sc.optimal = false
    fd_ref, pt_ref =
      fd_refinement(fd3, id, res, itd.Δxy, pt, itd, ϵ3, dda, pad, spd, cnts, iconf.mode)
    iter!(pt_ref, itd, fd_ref, id, res, sc, dda, pad, ϵ3, cnts, iconf, display)
    update_pt_ref!(fd_ref.Δref, pt, pt_ref, res, id, fd3, itd)
  end
  cnts.iters_sp3 = cnts.k - cnts.iters_sp2

  if iconf.scaling
    post_scale!(sd, pt, res, fd3, id, itd)
  end

  set_ripqp_stats!(stats, pt, res, pad, itd, id, sc, cnts, itol.max_iter)
  return stats
end


function SolverCore.solve!(
  solver::RipQPDoubleSolver{T},
  QM::AbstractQuadraticModel{T},
  stats::GenericExecutionStats{T};
) where {T}
  id = solver.id
  iconf, itol = solver.iconf, solver.itol
  sd, spd = solver.sd, solver.spd
  sc, cnts = solver.sc, solver.cnts
  display = solver.display
  sp, sp2 = solver.sp, solver.sp2
  solve_method, solve_method2 = solver.solve_method, solver.solve_method2
  pfd = solver.pfd
  pt, res, itd, dda, pad = pfd.pt, pfd.res, pfd.itd, pfd.dda, pfd.pad 
  fd1, ϵ1 = solver.fd1, solver.ϵ1
  fd2, ϵ2 = solver.fd2, solver.ϵ2
  cnts.time_solve = time()
  sc.max_iter = itol.max_iter1 # set max_iter for 1st solver
  cnts.last_sp = false # sp is not the last sp because sp2 is not nothing
  mode = iconf.mode

  T2 = solver_type(sp2)
  @assert T2 == T
  # initialize data
  initialize!(fd1, id, res, itd, dda, pad, pt, spd, ϵ1, sc, cnts, itol.max_time, display, T)
  if (iconf.mode == :multi || iconf.mode == :multizoom || iconf.mode == :multiref)
    set_tol_residuals!(ϵ2, T(res.rbNorm), T(res.rcNorm))
  end

  # IPM iterations 1st solver (mono mode: only this call to the iter! function)
  iter!(pt, itd, fd1, id, res, sc, dda, pad, ϵ1, cnts, iconf, display, last_iter = false)
  cnts.iters_sp = cnts.k

  sc.max_iter = itol.max_iter # set max_iter for 2nd solver
  cnts.last_sp = true # sp2 is the last sp because sp3 is nothing
  pt, itd, res, dda, pad =
    convert_types(T2, pt, itd, res, dda, pad, sp, sp2, id, fd2, solve_method, solve_method2)
  sc.optimal = itd.pdd < ϵ2.pdd && res.rbNorm < ϵ2.tol_rb && res.rcNorm < ϵ2.tol_rc
  sc.small_μ = itd.μ < ϵ2.μ
  display && show_used_solver(pad)
  display && setup_log_header(pad)

  if !sc.optimal && mode == :multi
    iter!(pt, itd, fd2, id, res, sc, dda, pad, ϵ2, cnts, iconf, display)
  elseif !sc.optimal && (mode == :multizoom || mode == :multiref)
    spd = convert(StartingPointData{T, typeof(pt.x)}, spd)
    fd_ref, pt_ref = fd_refinement(
      fd2,
      id,
      res,
      itd.Δxy,
      pt,
      itd,
      ϵ2,
      dda,
      pad,
      spd,
      cnts,
      iconf.mode,
      centering = true,
    )
    iter!(pt_ref, itd, fd_ref, id, res, sc, dda, pad, ϵ2, cnts, iconf, display)
    update_pt_ref!(fd_ref.Δref, pt, pt_ref, res, id, fd2, itd)
  elseif iconf.mode == :zoom || iconf.mode == :ref
    sc.optimal = false
    fd_ref, pt_ref =
      fd_refinement(fd2, id, res, itd.Δxy, pt, itd, ϵ2, dda, pad, spd, cnts, iconf.mode)
    iter!(pt_ref, itd, fd_ref, id, res, sc, dda, pad, ϵ2, cnts, iconf, display)
    update_pt_ref!(fd_ref.Δref, pt, pt_ref, res, id, fd2, itd)
  end
  cnts.iters_sp2 = cnts.k - cnts.iters_sp

  if iconf.scaling
    post_scale!(sd, pt, res, fd2, id, itd)
  end

  set_ripqp_stats!(stats, pt, res, pad, itd, id, sc, cnts, itol.max_iter)
  return stats
end

function SolverCore.solve!(
  solver::RipQPSolver{T},
  QM::AbstractQuadraticModel{T},
  stats::GenericExecutionStats{T};
) where {T}
  id = solver.id
  iconf, itol = solver.iconf, solver.itol
  sd, spd = solver.sd, solver.spd
  sc, cnts = solver.sc, solver.cnts
  display = solver.display
  pfd = solver.pfd
  pt, res, itd, dda, pad = pfd.pt, pfd.res, pfd.itd, pfd.dda, pfd.pad 
  fd, ϵ = solver.fd, solver.ϵ
  cnts.time_solve = time()

  # initialize data
  initialize!(fd, id, res, itd, dda, pad, pt, spd, ϵ, sc, cnts, itol.max_time, display, T)

  # IPM iterations 1st solver (mono mode: only this call to the iter! function)
  iter!(pt, itd, fd, id, res, sc, dda, pad, ϵ, cnts, iconf, display, last_iter = true)

  if iconf.scaling
    post_scale!(sd, pt, res, fd, id, itd)
  end

  set_ripqp_stats!(stats, pt, res, pad, itd, id, sc, cnts, itol.max_iter)
  return stats
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
  ps::Bool = true,
  scaling::Bool = true,
  history::Bool = false,
  mode::Symbol = :mono,
  sp::SolverParams = keep_init_prec(mode) ? K2LDLParams{T}() : K2LDLParams{Float32}(),
  display::Bool = true,
  kwargs...,
) where {T, S}
  start_time = time()
  # conversion function if QM.data.H and QM.data.A are not in the type required by sp
  QMi, ps, scaling = convert_QM(QM, ps, scaling, sp, display)
  if ps
    stats_ps = presolve(QMi, fixed_vars_only = !ps)
    if stats_ps.status == :unknown
      QMps = stats_ps.solver_specific[:presolvedQM]
    else
      return stats_ps
    end
  else
    QMps = QMi
  end
  psoperations = typeof(QMps) <: QuadraticModels.PresolvedQuadraticModel ? QMps.psd.operations : []
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
  solver = RipQPSolver(
    QM_inner;
    history = history,
    ps = ps,
    scaling = scaling,
    mode = mode,
    sp = sp,
    display = display,
    kwargs...,
  )
  SolverCore.solve!(solver, QM_inner, stats_inner)
  multipliers, multipliers_L, multipliers_U = get_slack_multipliers(
    stats_inner.multipliers,
    stats_inner.multipliers_L,
    stats_inner.multipliers_U,
    solver.id,
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
  solver_specific[:psoperations] = psoperations
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
