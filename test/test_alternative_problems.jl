@testset "LLS" begin
  # least-square problem without constraints
  m, n = 20, 10
  A = rand(m, n)
  b = rand(m)
  nls = LLSModel(A, b)
  statsnls = ripqp(nls, itol = InputTol(ϵ_rb = sqrt(eps()), ϵ_rc = sqrt(eps())), display = false)
  x, r = statsnls.solution, statsnls.solver_specific[:r]
  @test norm(x - A \ b) ≤ norm(b) * sqrt(eps()) * 100
  @test norm(A * x - b - r) ≤ norm(b) * sqrt(eps()) * 100

  # least-square problem with constraints
  A = rand(m, n)
  b = rand(m)
  lcon, ucon = zeros(m), fill(Inf, m)
  C = ones(m, n)
  lvar, uvar = fill(-10.0, n), fill(200.0, n)
  nls = LLSModel(A, b, lvar = lvar, uvar = uvar, C = C, lcon = lcon, ucon = ucon)
  statsnls = ripqp(nls, display = false)
  x, r = statsnls.solution, statsnls.solver_specific[:r]
  @test length(r) == m
  @test norm(A * x - b - r) ≤ norm(b) * sqrt(eps())
end

@testset "matrixwrite" begin
  stats1 = ripqp(
    QuadraticModel(qps1),
    display = false,
    w = SystemWrite(write = true, name = "test_", kfirst = 4, kgap = 1000),
  )
  @test isapprox(stats1.objective, -1.59078179, atol = 1e-2)
  @test stats1.status == :first_order

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
  @test stats1.status == :first_order

  stats2 = ripqp(QuadraticModelMaximize(qps2), ps = false, display = false)
  @test isapprox(stats2.objective, 9.99599999e1, atol = 1e-2)
  @test stats2.status == :first_order

  stats3 = ripqp(QuadraticModelMaximize(qps3), display = false)
  @test isapprox(stats3.objective, -5.32664756, atol = 1e-2)
  @test stats3.status == :first_order
end

@testset "presolve" begin
  qp = QuadraticModel(zeros(2), zeros(2, 2), lvar = zeros(2), uvar = zeros(2))
  stats_ps = ripqp(qp)
  @test stats_ps.status == :first_order
  @test stats_ps.solution == [0.0; 0.0]

  stats5 = ripqp(QuadraticModel(qps5), display = false)
  @test isapprox(stats5.objective, 0.250000001, atol = 1e-2)
  @test stats5.status == :first_order

  stats5 = ripqp(QuadraticModel(qps5), sp = K2KrylovParams(), display = false)
  @test isapprox(stats5.objective, 0.250000001, atol = 1e-2)
  @test stats5.status == :first_order
end

qp_linop = QuadraticModel(
  c,
  LinearOperator(Q),
  A = LinearOperator(A),
  lcon = b,
  ucon = b,
  lvar = l,
  uvar = u,
  c0 = 0.0,
  name = "QM_LINOP",
)
qp_dense = QuadraticModel(
  c,
  tril(Q),
  A = A,
  lcon = b,
  ucon = b,
  lvar = l,
  uvar = u,
  c0 = 0.0,
  name = "QM_LINOP",
)

@testset "Dense and LinearOperator QPs" begin
  stats_linop = ripqp(qp_linop, sp = K2KrylovParams(), ps = false, scaling = false, display = false)
  @test isapprox(stats_linop.objective, 1.1249999990782493, atol = 1e-2)
  @test stats_linop.status == :first_order

  stats_dense = ripqp(qp_dense, sp = K2KrylovParams(uplo = :U))
  @test isapprox(stats_dense.objective, 1.1249999990782493, atol = 1e-2)
  @test stats_dense.status == :first_order

  for fact_alg in [:bunchkaufman, :ldl]
    stats_dense = ripqp(
      qp_dense,
      sp = K2LDLDenseParams(fact_alg = fact_alg, ρ0 = 0.0, δ0 = 0.0),
      ps = false,
      scaling = false,
      display = false,
    )
    @test isapprox(stats_dense.objective, 1.1249999990782493, atol = 1e-2)
    @test stats_dense.status == :first_order
  end

  # test conversion
  stats_dense = ripqp(qp_dense)
  @test isapprox(stats_dense.objective, 1.1249999990782493, atol = 1e-2)
  @test stats_dense.status == :first_order
end
