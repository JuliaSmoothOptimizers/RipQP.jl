RipQPMonoParameters{T}(;
  sp::SolverParams = K2LDLParams{T}(),
  solve_method::SolveMethod = PC(),
) where {T} = RipQPMonoParameters(sp, solve_method)

RipQPSolver(QM::AbstractQuadraticModel{T}, ap::RipQPMonoParameters{T}; kwargs...) where {T} =
  RipQPMonoSolver(QM; ap = ap, kwargs...)

function RipQPMonoSolver(
  QM::AbstractQuadraticModel{T, S};
  itol::InputTol{T, Int} = InputTol(T),
  scaling::Bool = true,
  normalize_rtol::Bool = true,
  kc::I = 0,
  perturb::Bool = false,
  mode::Symbol = :mono,
  ap::RipQPMonoParameters{T} = RipQPMonoParameters{T}(),
  history::Bool = false,
  w::SystemWrite = SystemWrite(),
  display::Bool = true,
) where {T <: Real, S <: AbstractVector{T}, I <: Integer}
  start_time = time()
  elapsed_time = 0.0
  typeof(ap.solve_method) <: IPF &&
    kc != 0 &&
    error("IPF method should not be used with centrality corrections")
  iconf =
    InputConfig(mode, true, scaling, normalize_rtol, kc, perturb, QM.meta.minimize, history, w)

  # allocate workspace
  sc, fd, id, ϵ, res, itd, pt, sd, spd, cnts =
    allocate_workspace(QM, iconf, itol, start_time, T, ap.sp)

  if iconf.scaling
    scaling!(fd, id, sd, T(1.0e-5))
  end

  dda = DescentDirectionAllocs(id, ap.solve_method, S)

  pad = PreallocatedData(ap.sp, fd, id, itd, pt, iconf)

  pfd = PreallocatedFloatData(pt, res, itd, dda, pad)

  cnts.time_allocs = time() - start_time

  return RipQPMonoSolver(QM, id, iconf, itol, sd, spd, sc, cnts, display, fd, ϵ, pfd)
end

function SolverCore.solve!(
  solver::RipQPMonoSolver{T, S},
  QM::AbstractQuadraticModel{T, S},
  stats::GenericExecutionStats{T, S},
  ap::RipQPMonoParameters{T},
) where {T, S}
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
