@testset "Krylov K1" begin
  for kmethod in [:cg, :cg_lanczos, :cr]
    stats4 = ripqp(
      QuadraticModel(qps4),
      display = false,
      sp = K1KrylovParams(kmethod = kmethod),
      solve_method = IPF(),
      history = true,
      itol = InputTol(max_iter = 50, max_time = 20.0, ϵ_rc = 1.0e-6, ϵ_rb = 1.0e-6, ϵ_pdd = 1.0e-6),
    )
    @test isapprox(stats4.objective, -4.6475314286e02, atol = 1e-2)
    @test stats4.status == :first_order
  end

  stats4 = ripqp(
    QuadraticModel(qps4),
    display = false,
    sp = K1KrylovParams(uplo = :U),
    solve_method = PC(),
    history = true,
    itol = InputTol(max_iter = 50, max_time = 20.0, ϵ_rc = 1.0e-6, ϵ_rb = 1.0e-6, ϵ_pdd = 1.0e-6),
  )
  @test isapprox(stats4.objective, -4.6475314286e02, atol = 1e-2)
  @test stats4.status == :first_order
end

@testset "Krylov K1.1 Structured" begin
  for kmethod in [:lsqr, :lsmr]
    stats4 = ripqp(
      QuadraticModel(qps4),
      display = false,
      sp = K1_1StructuredParams(kmethod = kmethod),
      solve_method = IPF(),
      history = true,
      itol = InputTol(max_iter = 50, max_time = 20.0, ϵ_rc = 1.0e-6, ϵ_rb = 1.0e-6, ϵ_pdd = 1.0e-6),
    )
    @test isapprox(stats4.objective, -4.6475314286e02, atol = 1e-1)
    @test stats4.status == :first_order
  end

  stats4 = ripqp(
    QuadraticModel(qps4),
    display = false,
    sp = K1_1StructuredParams(uplo = :U),
    solve_method = IPF(),
    history = true,
    itol = InputTol(max_iter = 50, max_time = 20.0, ϵ_rc = 1.0e-6, ϵ_rb = 1.0e-6, ϵ_pdd = 1.0e-6),
  )
  @test isapprox(stats4.objective, -4.6475314286e02, atol = 1e-2)
  @test stats4.status == :first_order
end

@testset "Krylov K1.2 Structured" begin
  for kmethod in [:lnlq, :craig, :craigmr]
    stats4 = ripqp(
      QuadraticModel(qps4),
      display = false,
      sp = K1_2StructuredParams(kmethod = kmethod),
      solve_method = IPF(),
      history = true,
      itol = InputTol(max_iter = 50, max_time = 20.0, ϵ_rc = 1.0e-6, ϵ_rb = 1.0e-6, ϵ_pdd = 1.0e-6),
    )
    @test isapprox(stats4.objective, -4.6475314286e02, atol = 1e-1)
    @test stats4.status == :first_order
  end

  stats4 = ripqp(
    QuadraticModel(qps4),
    display = false,
    sp = K1_2StructuredParams(uplo = :U),
    solve_method = IPF(),
    history = true,
    itol = InputTol(max_iter = 50, max_time = 20.0, ϵ_rc = 1.0e-6, ϵ_rb = 1.0e-6, ϵ_pdd = 1.0e-6),
  )
  @test isapprox(stats4.objective, -4.6475314286e02, atol = 1e-2)
  @test stats4.status == :first_order
end

@testset "Krylov K1 Dense" begin
  sm_dense4 = SlackModel(QuadraticModel(qps4))
  qm_dense4 = QuadraticModel(
    sm_dense4.data.c,
    Matrix(sm_dense4.data.H),
    A = Matrix(sm_dense4.data.A),
    lcon = sm_dense4.meta.lcon,
    ucon = sm_dense4.meta.ucon,
    lvar = sm_dense4.meta.lvar,
    uvar = sm_dense4.meta.uvar,
  )
  stats4 = ripqp(
    qm_dense4,
    display = false,
    sp = K1CholDenseParams(),
    solve_method = PC(),
    ps = false,
    scaling = false,
    history = false,
    itol = InputTol(max_iter = 50, max_time = 20.0, ϵ_rc = 1.0e-6, ϵ_rb = 1.0e-6, ϵ_pdd = 1.0e-6),
  )

  @test isapprox(stats4.objective, -4.6475314286e02, atol = 1e-2)
  @test stats4.status == :first_order

  stats4 = ripqp(
    qm_dense4,
    display = true,
    sp = K1CholDenseParams(),
    solve_method = PC(),
    mode = :multi,
    ps = false,
    scaling = false,
    history = false,
    itol = InputTol(max_iter = 50, max_time = 20.0, ϵ_rc = 1.0e-6, ϵ_rb = 1.0e-6, ϵ_pdd = 1.0e-6),
  )

  @test isapprox(stats4.objective, -4.6475314286e02, atol = 1e-2)
  @test stats4.status == :first_order
end
