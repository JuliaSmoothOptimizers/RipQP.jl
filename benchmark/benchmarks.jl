using BenchmarkTools
using RipQP
using QPSReader
using QuadraticModels
using Logging


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

const SUITE = BenchmarkGroup()
SUITE[:ripqp] = BenchmarkGroup()


for (i,nlp) in enumerate(problems)
  SUITE[:ripqp]["problem#$i"] = @benchmarkable ripqp($nlp;display=false)
end
