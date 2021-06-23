using RipQP, LLSModels, QPSReader, QuadraticModels, Quadmath
using DelimitedFiles, LinearAlgebra, MatrixMarket, Test

function createQuadraticModel128(qpdata; name = "qp_pb")
  return QuadraticModel(
    convert(Array{Float128}, qpdata.c),
    qpdata.qrows,
    qpdata.qcols,
    convert(Array{Float128}, qpdata.qvals),
    Arows = qpdata.arows,
    Acols = qpdata.acols,
    Avals = convert(Array{Float128}, qpdata.avals),
    lcon = convert(Array{Float128}, qpdata.lcon),
    ucon = convert(Array{Float128}, qpdata.ucon),
    lvar = convert(Array{Float128}, qpdata.lvar),
    uvar = convert(Array{Float128}, qpdata.uvar),
    c0 = Float128(qpdata.c0),
    x0 = zeros(Float128, length(qpdata.c)),
    name = name,
  )
end

@testset "mono_mode" begin
  qps1 = readqps("QAFIRO.SIF") #lower bounds
  stats1 = ripqp(QuadraticModel(qps1))
  @test isapprox(stats1.objective, -1.59078179, atol = 1e-2)
  @test stats1.status == :acceptable

  qps2 = readqps("HS21.SIF") # low/upp bounds
  stats2 = ripqp(QuadraticModel(qps2))
  @test isapprox(stats2.objective, -9.99599999e1, atol = 1e-2)
  @test stats1.status == :acceptable

  qps3 = readqps("HS52.SIF") # free bounds
  stats3 = ripqp(QuadraticModel(qps3))
  @test isapprox(stats3.objective, 5.32664756, atol = 1e-2)
  @test stats3.status == :acceptable
end

@testset "multi_mode" begin
  qps1 = readqps("QAFIRO.SIF") #lower bounds
  stats1 = ripqp(QuadraticModel(qps1), iconf = InputConfig(mode = :multi), display = false)
  @test isapprox(stats1.objective, -1.59078179, atol = 1e-2)
  @test stats1.status == :acceptable

  qps2 = readqps("HS21.SIF") # low/upp bounds
  stats2 = ripqp(QuadraticModel(qps2), iconf = InputConfig(mode = :multi), display = false)
  @test isapprox(stats2.objective, -9.99599999e1, atol = 1e-2)
  @test stats2.status == :acceptable

  qps3 = readqps("HS52.SIF") # free bounds
  stats3 = ripqp(QuadraticModel(qps3), iconf = InputConfig(mode = :multi), display = false)
  @test isapprox(stats3.objective, 5.32664756, atol = 1e-2)
  @test stats3.status == :acceptable
end

@testset "dynamic_regularization" begin
  qps1 = readqps("QAFIRO.SIF") #lower bounds
  stats1 = ripqp(
    QuadraticModel(qps1),
    iconf = InputConfig(sp = K2LDLParams(regul = :dynamic)),
    display = false,
  )
  @test isapprox(stats1.objective, -1.59078179, atol = 1e-2)
  @test stats1.status == :acceptable

  qps2 = readqps("HS21.SIF") # low/upp bounds
  stats2 = ripqp(
    QuadraticModel(qps2),
    iconf = InputConfig(sp = K2LDLParams(regul = :dynamic)),
    display = false,
  )
  @test isapprox(stats2.objective, -9.99599999e1, atol = 1e-2)
  @test stats2.status == :acceptable

  qps3 = readqps("HS52.SIF") # free bounds
  stats3 = ripqp(
    QuadraticModel(qps3),
    iconf = InputConfig(sp = K2LDLParams(regul = :dynamic)),
    display = false,
  )
  @test isapprox(stats3.objective, 5.32664756, atol = 1e-2)
  @test stats3.status == :acceptable
end

@testset "centrality_corrections" begin
  qps1 = readqps("QAFIRO.SIF") #lower bounds
  stats1 = ripqp(QuadraticModel(qps1), iconf = InputConfig(kc = -1), display = false) # automatic centrality corrections computation
  @test isapprox(stats1.objective, -1.59078179, atol = 1e-2)
  @test stats1.status == :acceptable

  qps2 = readqps("HS21.SIF") # low/upp bounds
  stats2 = ripqp(QuadraticModel(qps2), iconf = InputConfig(kc = 2), display = false)
  @test isapprox(stats2.objective, -9.99599999e1, atol = 1e-2)
  @test stats2.status == :acceptable

  qps3 = readqps("HS52.SIF") # free bounds
  stats3 = ripqp(QuadraticModel(qps3), iconf = InputConfig(kc = 2), display = false)
  @test isapprox(stats3.objective, 5.32664756, atol = 1e-2)
  @test stats3.status == :acceptable
end

@testset "Float128" begin
  qps1 = readqps("QAFIRO.SIF") #lower bounds
  qm128_1 = createQuadraticModel128(qps1)
  stats1 = ripqp(
    qm128_1,
    itol = InputTol(ϵ_rb32 = 0.1, ϵ_rb64 = 0.01),
    iconf = InputConfig(mode = :multi, normalize_rtol = false),
    display = false,
  )
  @test isapprox(stats1.objective, -1.59078179, atol = 1e-2)
  @test stats1.status == :acceptable

  qps2 = readqps("HS21.SIF") # low/upp bounds
  qm128_2 = createQuadraticModel128(qps2)
  stats2 = ripqp(qm128_2, display = false)
  @test isapprox(stats2.objective, -9.99599999e1, atol = 1e-2)
  @test stats2.status == :acceptable
end

@testset "refinement" begin
  qps1 = readqps("QAFIRO.SIF") #lower bounds
  stats1 = ripqp(QuadraticModel(qps1), iconf = InputConfig(refinement = :zoom), display = false) # automatic centrality corrections computation
  @test isapprox(stats1.objective, -1.59078179, atol = 1e-2)
  @test stats1.status == :acceptable

  qps1 = readqps("QAFIRO.SIF") #lower bounds
  stats1 = ripqp(QuadraticModel(qps1), iconf = InputConfig(refinement = :ref), display = false) # automatic centrality corrections computation
  @test isapprox(stats1.objective, -1.59078179, atol = 1e-2)
  @test stats1.status == :acceptable

  qps2 = readqps("HS21.SIF") # low/upp bounds
  stats2 = ripqp(
    QuadraticModel(qps2),
    iconf = InputConfig(mode = :multi, refinement = :multizoom),
    display = false,
  )
  @test isapprox(stats2.objective, -9.99599999e1, atol = 1e-2)
  @test stats2.status == :acceptable

  qps3 = readqps("HS52.SIF") # free bounds
  stats3 = ripqp(
    QuadraticModel(qps3),
    iconf = InputConfig(mode = :multi, refinement = :multiref),
    display = false,
  )
  @test isapprox(stats3.objective, 5.32664756, atol = 1e-2)
  @test stats3.status == :acceptable
end

@testset "K2_5" begin
  qps1 = readqps("QAFIRO.SIF") #lower bounds
  stats1 = ripqp(QuadraticModel(qps1), display = false, iconf = InputConfig(sp = K2_5LDLParams()))
  @test isapprox(stats1.objective, -1.59078179, atol = 1e-2)
  @test stats1.status == :acceptable

  qps2 = readqps("HS21.SIF") # low/upp bounds
  stats2 = ripqp(
    QuadraticModel(qps2),
    display = false,
    iconf = InputConfig(sp = K2_5LDLParams(), mode = :multi),
  )
  @test isapprox(stats2.objective, -9.99599999e1, atol = 1e-2)
  @test stats2.status == :acceptable

  qps3 = readqps("HS52.SIF") # free bounds
  stats3 = ripqp(
    QuadraticModel(qps3),
    display = false,
    iconf = InputConfig(sp = K2_5LDLParams(regul = :dynamic)),
  )
  @test isapprox(stats3.objective, 5.32664756, atol = 1e-2)
  @test stats3.status == :acceptable
end

@testset "IPF" begin
  qps1 = readqps("QAFIRO.SIF") #lower bounds
  stats1 = ripqp(
    QuadraticModel(qps1),
    display = false,
    iconf = InputConfig(solve_method = :IPF, mode = :multi),
  )
  @test isapprox(stats1.objective, -1.59078179, atol = 1e-2)
  @test stats1.status == :acceptable

  qps2 = readqps("HS21.SIF") # low/upp bounds
  stats2 = ripqp(QuadraticModel(qps2), display = false, iconf = InputConfig(solve_method = :IPF))
  @test isapprox(stats2.objective, -9.99599999e1, atol = 1e-2)
  @test stats2.status == :acceptable

  qps2 = readqps("HS21.SIF") # low/upp bounds
  stats2 = ripqp(
    QuadraticModel(qps2),
    display = false,
    iconf = InputConfig(solve_method = :IPF, sp = K2_5LDLParams(), refinement = :zoom),
  )
  @test isapprox(stats2.objective, -9.99599999e1, atol = 1e-2)
  @test stats2.status == :acceptable

  qps1 = readqps("QAFIRO.SIF") #lower bounds
  stats1 = ripqp(
    QuadraticModel(qps1),
    display = false,
    iconf = InputConfig(solve_method = :IPF, mode = :multi, refinement = :multiref),
  )
  @test isapprox(stats1.objective, -1.59078179, atol = 1e-2)
  @test stats1.status == :acceptable

  qps2 = readqps("HS21.SIF") # low/upp bounds
  stats2 = ripqp(
    QuadraticModel(qps2),
    display = false,
    iconf = InputConfig(
      solve_method = :IPF,
      mode = :multi,
      refinement = :multizoom,
      sp = K2_5LDLParams(),
    ),
  )
  @test isapprox(stats2.objective, -9.99599999e1, atol = 1e-2)
  @test stats2.status == :acceptable

  qps2 = readqps("HS21.SIF") # low/upp bounds
  stats2 = ripqp(
    QuadraticModel(qps2),
    display = false,
    iconf = InputConfig(
      solve_method = :IPF,
      refinement = :zoom,
      sp = K2LDLParams(regul = :dynamic),
    ),
  )
  @test isapprox(stats2.objective, -9.99599999e1, atol = 1e-2)
  @test stats2.status == :acceptable
end

@testset "LLS" begin
  # least-square problem without constraints
  m, n = 20, 10
  A = rand(m, n)
  b = rand(m)
  nls = LLSModel(A, b)
  statsnls = ripqp(nls, itol = InputTol(ϵ_rb = sqrt(eps()), ϵ_rc = sqrt(eps())), display = false)
  x, r = statsnls.solution[1:n], statsnls.solution[(n + 1):end]
  @test norm(x - A \ b) ≤ norm(b) * sqrt(eps())
  @test norm(A * x - b - r) ≤ norm(b) * sqrt(eps())

  # least-square problem with constraints
  A = rand(m, n)
  b = rand(m)
  lcon, ucon = zeros(m), fill(Inf, m)
  C = ones(m, n)
  lvar, uvar = fill(-10.0, n), fill(200.0, n)
  nls = LLSModel(A, b, lvar = lvar, uvar = uvar, C = C, lcon = lcon, ucon = ucon)
  statsnls = ripqp(nls, display = false)
  x, r = statsnls.solution[1:n], statsnls.solution[(n + 1):end]
  @test length(r) == m
  @test norm(A * x - b - r) ≤ norm(b) * sqrt(eps())
end

@testset "matrixwrite" begin
  qps1 = readqps("QAFIRO.SIF") #lower bounds
  stats1 = ripqp(
    QuadraticModel(qps1),
    display = false,
    iconf = InputConfig(w = SystemWrite(write = true, name = "test_", kfirst = 4, kgap = 1000)),
  )
  @test isapprox(stats1.objective, -1.59078179, atol = 1e-2)
  @test stats1.status == :acceptable

  K = MatrixMarket.mmread("test_K_iter4.mtx")
  rhs_aff = readdlm("test_rhs_iter4_aff.rhs", Float64)[:]
  rhs_cc = readdlm("test_rhs_iter4_cc.rhs", Float64)[:]
  @test size(K, 1) == size(K, 2) == length(rhs_aff) == length(rhs_cc)
end
