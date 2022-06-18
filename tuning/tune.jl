using Pkg
using Distributed
using QPSReader
using Logging
using NamedTupleTools
# using QuadraticModels

# 1. Launch workers
# init_workers(; exec_flags="--project=$(@__DIR__)")

# 2. make modules available to all workers:
@everywhere begin
  using RipQP,
  SolverTuning,
  BBModels,
  Statistics,
  NLPModels,
  QuadraticModels
end

T=Float64
function createQuadraticModelT(qpdata; T = Float64, name = "qp_pb")
  return QuadraticModel(
    convert(Array{T}, qpdata.c),
    qpdata.qrows,
    qpdata.qcols,
    convert(Array{T}, qpdata.qvals),
    Arows = qpdata.arows,
    Acols = qpdata.acols,
    Avals = convert(Array{T}, qpdata.avals),
    lcon = convert(Array{T}, qpdata.lcon),
    ucon = convert(Array{T}, qpdata.ucon),
    lvar = convert(Array{T}, qpdata.lvar),
    uvar = convert(Array{T}, qpdata.uvar),
    c0 = T(qpdata.c0),
    x0 = zeros(T, length(qpdata.c)),
    name = name,
  )
end

# 3. Setup problems
# netlib_problem_path = [joinpath(fetch_netlib(), path) for path ∈ readdir(fetch_netlib()) if match(r".*\.(SIF|QPS)$", path) !== nothing]
mm_problem_path = [joinpath(fetch_mm(), path) for path ∈ readdir(fetch_mm()) if match(r".*\.(SIF|QPS)$", path) !== nothing]

# problem_paths = cat(mm_problem_path, netlib_problem_path; dims=1)
problem_paths = mm_problem_path

problems = QuadraticModel[]
for (i,problem_path) ∈ enumerate(problem_paths)
  try
    with_logger(NullLogger()) do
      qps = readqps(problem_path)
      p = createQuadraticModelT(qps;name="qp_pb_$i")
      push!(problems, p)
    end
  catch e
    @warn e
  end
end

x = default_parameters(T)
x = delete(x, :kc)

# 5. Define a BBModel problem:

# 5.1 define a function that executes your solver. It must take an nlp followed by a vector of real values:
@everywhere function solver_func(nlp::AbstractNLPModel, p::NamedTuple)
  return ripqp(nlp, p; display=false)
end

# 5.2 Define a function that takes a ProblemMetric object. This function must return one real number.

@everywhere function aux_func(p_metric::ProblemMetrics)
  median_time = median(get_times(p_metric))
  memory = get_memory(p_metric)
  solved = get_solved(p_metric)
  counters = get_counters(p_metric)

  return median_time + memory + counters.neval_obj + (Float64(!solved) * 5.0 * median_time)
end

lvar=[T(0.8), T(0)+eps(T)]
uvar=[T(1)-eps(T), T(0.2)]

bbmodel = BBModel(x, solver_func, aux_func, problems;lvar=lvar, uvar=uvar)

best_params, param_opt_problem = solve_bb_model(bbmodel;lb_choice=:C,
display_all_eval = true,
# max_time = 300,
# max_bb_eval =600,
display_stats = ["BBE", "SOL", "CONS_H", "TIME", "OBJ"],
)