
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

function QuadraticModelMaximize(qps, x0 = zeros(qps.nvar))
  QuadraticModel(
    -qps.c,
    qps.qrows,
    qps.qcols,
    -qps.qvals,
    Arows = qps.arows,
    Acols = qps.acols,
    Avals = qps.avals,
    lcon = qps.lcon,
    ucon = qps.ucon,
    lvar = qps.lvar,
    uvar = qps.uvar,
    c0 = -qps.c0,
    x0 = x0,
    minimize = false,
  )
end

@testset "maximize" begin
  stats1 = ripqp(QuadraticModelMaximize(qps1), display = false)
  @test isapprox(stats1.objective, 1.59078179, atol = 1e-2)
  @test stats1.status == :acceptable

  stats2 = ripqp(QuadraticModelMaximize(qps2), display = false)
  @test isapprox(stats2.objective, 9.99599999e1, atol = 1e-2)
  @test stats2.status == :acceptable

  stats3 = ripqp(QuadraticModelMaximize(qps3), display = false)
  @test isapprox(stats3.objective, -5.32664756, atol = 1e-2)
  @test stats3.status == :acceptable
end

@testset "presolve" begin
  stats5 = ripqp(QuadraticModel(qps5), display = false)
  @test isapprox(stats5.objective, 0.250000001, atol = 1e-2)
  @test stats5.status == :acceptable

  stats5 = ripqp(QuadraticModel(qps5), iconf = InputConfig(sp = K2KrylovParams()), display = false)
  @test isapprox(stats5.objective, 0.250000001, atol = 1e-2)
  @test stats5.status == :acceptable
end