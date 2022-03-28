using Pkg
using Distributed
using SolverParameters
using SolverTuning
using SolverCore
using NLPModels
using BenchmarkTools
using QPSReader
using Logging
# using QuadraticModels

# 1. Launch workers
init_workers(;nb_nodes=20, exec_flags="--project=$(@__DIR__)")

# 2. make modules available to all workers:
@everywhere begin
  using RipQP,
  SolverTuning,
  NLPModels,
  QuadraticModels
end

# 3. Setup problems
netlib_problem_path = [joinpath(fetch_netlib(), path) for path ∈ readdir(fetch_netlib()) if match(r".*\.(SIF|QPS)$", path) !== nothing]
# mm_problem_path = [joinpath(fetch_mm(), path) for path ∈ readdir(fetch_mm()) if match(r".*\.(SIF|QPS)$", path) !== nothing]

# problem_paths = cat(mm_problem_path, netlib_problem_path; dims=1)
problem_paths = netlib_problem_path

problems = QuadraticModel[]
for problem_path ∈ problem_paths
  try
    with_logger(NullLogger()) do
      qps = readqps(problem_path)
      p = QuadraticModel(qps)
      push!(problems, p)
    end
  catch e
    @warn e
  end
end

# Function that will count failures
function count_failures(bmark_results::Dict{Int, Float64}, stats_results::Dict{Int, AbstractExecutionStats})
  failure_penalty = 0.0
  for (pb_id, stats) in stats_results
    is_failure(stats) || continue
    failure_penalty += 25.0 * bmark_results[pb_id]
  end
  return failure_penalty
end

function is_failure(stats::AbstractExecutionStats)
  failure_status = [:exception, :infeasible, :max_eval, :max_iter, :max_time, :stalled, :neg_pred]
  return any(s -> s == stats.status, failure_status)
end

# 5. define user's blackbox:
# scaling_param = AlgorithmicParameter(true, BinaryRange(), "scaling")
# kc_param = AlgorithmicParameter(-1, IntegerRange(-1, 10), "kc")
# presolve_param = AlgorithmicParameter(true, BinaryRange(), "presolve")
# ripqp_params = [scaling_param, kc_param, presolve_param]
solver = RipQPSolver(first(problems))
kwargs = Dict{Symbol, Any}(:display => false)

function my_black_box(args...;kwargs...)
  # a little hack...
  bmark_results, stats_results, solver_results = eval_solver(ripqp, args...;kwargs...)
  bmark_results = Dict(pb_id => (median(bmark).time/1.0e9) for (pb_id, bmark) ∈ bmark_results)
  total_time = sum(values(bmark_results))
  failure_penalty = count_failures(bmark_results, stats_results)
  bb_result = total_time + failure_penalty
  @info "failure_penalty: $failure_penalty"

  return [bb_result], bmark_results, stats_results
end
black_box = BlackBox(solver, collect(values(solver.parameters)), my_black_box, kwargs)

# 7. define problem suite
param_optimization_problem =
  ParameterOptimizationProblem(black_box, problems; is_load_balanced=true)

# named arguments are options to pass to Nomad
create_nomad_problem!(
  param_optimization_problem;
  display_all_eval = true,
  max_time = 18000,
  max_bb_eval =200,
  display_stats = ["BBE", "EVAL", "SOL", "OBJ"],
)

# 8. Execute Nomad
result = solve_with_nomad!(param_optimization_problem)
@info ("Best feasible parameters: $(result.x_best_feas)")

for p ∈ black_box.solver_params
  @info "$(name(p)): $(default(p))"
end

rmprocs(workers())
