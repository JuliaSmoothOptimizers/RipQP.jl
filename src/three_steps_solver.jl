RipQPTripleParameters{T}(;
  sp::SolverParams = K2LDLParams{Float32}(),
  sp2::SolverParams = K2LDLParams{Float64}(),
  sp3::SolverParams{T} = K2LDLParams{T}(),
  solve_method::SolveMethod = PC(),
  solve_method2::SolveMethod = PC(),
  solve_method3::SolveMethod = PC(),
) where {T} = RipQPTripleParameters(sp, sp2, sp3, solve_method, solve_method2, solve_method3)

RipQPSolver(QM::AbstractQuadraticModel{T}, ap::RipQPTripleParameters{T}; kwargs...) where {T} =
  RipQPTripleSolver(QM; ap = ap, kwargs...)

function RipQPTripleSolver(
  QM::AbstractQuadraticModel{T0, S0};
  itol::InputTol{T0, Int} = InputTol(T0),
  scaling::Bool = true,
  normalize_rtol::Bool = true,
  kc::I = 0,
  perturb::Bool = false,
  mode::Symbol = :multi,
  early_multi_stop::Bool = true,
  ap::RipQPTripleParameters{T0} = RipQPTripleParameters{T0}(),
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
  if mode == :mono
    @warn "setting mode to multi because sp2 is provided"
    mode = :multi
  end
  typeof(ap.solve_method) <: IPF &&
    kc != 0 &&
    error("IPF method should not be used with centrality corrections")
  T = solver_type(ap.sp) # initial solve type
  iconf = InputConfig(
    mode,
    early_multi_stop,
    scaling,
    normalize_rtol,
    kc,
    perturb,
    QM.meta.minimize,
    history,
    w,
  )

  # allocate workspace
  sc, fd_T0, id, ϵ_T0, res, itd, pt, sd, spd, cnts =
    allocate_workspace(QM, iconf, itol, start_time, T0, ap.sp)

  if iconf.scaling
    scaling!(fd_T0, id, sd, T0(1.0e-5))
  end

  # extra workspace
  T2 = next_type(T, T0) # eltype of sp2
  fd, ϵ = allocate_extra_workspace1(T, itol, iconf, fd_T0)
  fd2, ϵ2 = allocate_extra_workspace2(T2, itol, iconf, fd_T0)
  
  S = change_vector_eltype(S0, T)
  dda = DescentDirectionAllocs(id, ap.solve_method, S)

  pad = PreallocatedData(ap.sp, fd, id, itd, pt, iconf)

  pfd = PreallocatedFloatData(pt, res, itd, dda, pad)

  cnts.time_allocs = time() - start_time

  return RipQPTripleSolver(
    QM,
    id,
    iconf,
    itol,
    sd,
    spd,
    sc,
    cnts,
    display,
    fd,
    ϵ,
    fd2,
    ϵ2,
    fd_T0,
    ϵ_T0,
    pfd,
  )
end

function SolverCore.solve!(
  solver::RipQPTripleSolver{T, S},
  QM::AbstractQuadraticModel{T, S},
  stats::GenericExecutionStats{T, S},
  ap::RipQPTripleParameters,
) where {T, S}
  id = solver.id
  iconf, itol = solver.iconf, solver.itol
  sd, spd = solver.sd, solver.spd
  sc, cnts = solver.sc, solver.cnts
  display = solver.display
  sp, sp2, sp3 = ap.sp, ap.sp2, ap.sp3
  solve_method, solve_method2, solve_method3 = ap.solve_method, ap.solve_method2, ap.solve_method3
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
  if (mode == :multi || mode == :multizoom || mode == :multiref)
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
      mode,
      centering = true,
    )
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
