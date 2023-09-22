@testset "mono_mode" begin
  stats1 = ripqp(QuadraticModel(qps1), ps = false)
  @test isapprox(stats1.objective, -1.59078179, atol = 1e-2)
  @test stats1.status == :first_order

  stats2 = ripqp(QuadraticModel(qps2), sp = K2LDLParams(fact_alg = CholmodFact()), display = false)
  @test isapprox(stats2.objective, -9.99599999e1, atol = 1e-2)
  @test stats1.status == :first_order

  if LIBHSL_isfunctional()
    stats2 =
      ripqp(QuadraticModel(qps2), sp = K2LDLParams(fact_alg = HSLMA57Fact()), display = false)
    @test isapprox(stats2.objective, -9.99599999e1, atol = 1e-2)
    @test stats1.status == :first_order
  end

  if LIBHSL_isfunctional()
    stats2 =
      ripqp(QuadraticModel(qps2), sp = K2LDLParams(fact_alg = HSLMA97Fact()), display = false)
    @test isapprox(stats2.objective, -9.99599999e1, atol = 1e-2)
    @test stats1.status == :first_order
  end

  stats3 = ripqp(QuadraticModel(qps3), display = false)
  @test isapprox(stats3.objective, 5.32664756, atol = 1e-2)
  @test stats3.status == :first_order

  stats4 = ripqp(QuadraticModel(qps4), display = false)
  @test isapprox(stats4.objective, -4.6475314286e02, atol = 1e-2)
  @test stats4.status == :first_order
end

@testset "K2_5" begin
  stats1 = ripqp(QuadraticModel(qps1), display = false, sp = K2_5LDLParams())
  @test isapprox(stats1.objective, -1.59078179, atol = 1e-2)
  @test stats1.status == :first_order

  stats2 = ripqp(QuadraticModel(qps2), display = false, sp = K2_5LDLParams(), mode = :multi)
  @test isapprox(stats2.objective, -9.99599999e1, atol = 1e-2)
  @test stats2.status == :first_order

  stats3 = ripqp(
    QuadraticModel(qps3),
    display = false,
    sp = K2_5LDLParams(fact_alg = LDLFact(regul = :dynamic)),
  )
  @test isapprox(stats3.objective, 5.32664756, atol = 1e-2)
  @test stats3.status == :first_order
end

@testset "KrylovK2" begin
  for precond in [Identity(), Jacobi(), Equilibration()]
    for kmethod in [:minres, :minres_qlp, :symmlq]
      stats2 = ripqp(
        QuadraticModel(qps2),
        display = false,
        sp = K2KrylovParams(uplo = :L, kmethod = kmethod, preconditioner = precond),
        history = true,
        itol = InputTol(
          max_iter = 50,
          max_time = 40.0,
          ϵ_rc = 1.0e-3,
          ϵ_rb = 1.0e-3,
          ϵ_pdd = 1.0e-3,
        ),
      )
      @test isapprox(stats2.objective, -9.99599999e1, atol = 1e-1)
      @test stats2.status == :first_order
    end
  end

  stats3 = ripqp(
    QuadraticModel(qps3),
    display = false,
    sp = K2KrylovParams(
      uplo = :U,
      kmethod = :minres_qlp,
      preconditioner = Equilibration(),
      form_mat = true,
    ),
    itol = InputTol(max_iter = 50, max_time = 40.0, ϵ_rc = 1.0e-2, ϵ_rb = 1.0e-2, ϵ_pdd = 1.0e-2),
  )
  @test isapprox(stats3.objective, 5.32664756, atol = 1e-1)
  @test stats3.status == :first_order

  for T in [Float32, Float64]
    for precond in
        [LDL(T = T, pos = :C), LDL(T = T, warm_start = false, pos = :L), LDL(T = T, pos = :R)]
      stats2 = ripqp(
        QuadraticModel(qps2),
        display = false,
        sp = K2KrylovParams(
          uplo = :U,
          kmethod = :gmres,
          preconditioner = precond,
          rhs_scale = true,
          form_mat = true,
          equilibrate = true,
        ),
        solve_method = IPF(),
      )
      @test isapprox(stats2.objective, -9.99599999e1, atol = 1e-1)
      @test stats2.status == :first_order
    end
  end

  stats3 = ripqp(
    QuadraticModel(qps3),
    display = true,
    sp = K2KrylovParams(
      uplo = :U,
      kmethod = :minres,
      preconditioner = LDL(),
      rhs_scale = true,
      form_mat = true,
      equilibrate = true,
    ),
    solve_method = IPF(),
  )
  @test isapprox(stats3.objective, 5.32664756, atol = 1e-1)
  @test stats3.status == :first_order

  stats3 = ripqp(
    QuadraticModel(qps3),
    display = false,
    sp = K2KrylovParams(
      uplo = :L,
      kmethod = :minres,
      preconditioner = LDL(fact_alg = LLDLFact()),
      rhs_scale = true,
      form_mat = true,
      equilibrate = true,
    ),
    solve_method = IPF(),
  )
  @test isapprox(stats3.objective, 5.32664756, atol = 1e-1)
  @test stats3.status == :first_order
end

@testset "KrylovK2_5" begin
  for kmethod in [:minres, :dqgmres]
    stats1 = ripqp(
      QuadraticModel(qps1),
      display = false,
      sp = K2_5KrylovParams(kmethod = kmethod, preconditioner = Identity()),
      solve_method = IPF(),
      history = true,
      itol = InputTol(max_iter = 50, max_time = 20.0, ϵ_rc = 1.0e-2, ϵ_rb = 1.0e-2, ϵ_pdd = 1.0e-2),
    )
    @test isapprox(stats1.objective, -1.59078179, atol = 1e-1)
    @test stats1.status == :first_order

    stats3 = ripqp(
      QuadraticModel(qps3),
      display = false,
      sp = K2_5KrylovParams(kmethod = kmethod, preconditioner = Jacobi()),
      itol = InputTol(max_iter = 50, max_time = 20.0, ϵ_rc = 1.0e-2, ϵ_rb = 1.0e-2, ϵ_pdd = 1.0e-2),
    )
    @test isapprox(stats3.objective, 5.32664756, atol = 1e-1)
    @test stats3.status == :first_order
  end
end

@testset "K2 structured LP" begin
  for kmethod in [:gpmr, :trimr]
    for δ_min in [1.0e-2, 0.0]
      stats4 = ripqp(
        QuadraticModel(qps4),
        display = false,
        sp = K2StructuredParams(kmethod = kmethod, δ_min = δ_min),
        solve_method = IPF(),
        itol = InputTol(
          max_iter = 50,
          max_time = 20.0,
          ϵ_rc = 1.0e-4,
          ϵ_rb = 1.0e-4,
          ϵ_pdd = 1.0e-4,
        ),
      )
      @test isapprox(stats4.objective, -4.6475314286e02, atol = 1e-1)
      @test stats4.status == :first_order
    end
  end
  stats4 = ripqp(
    QuadraticModel(qps4),
    display = false,
    sp = K2StructuredParams(kmethod = :tricg),
    solve_method = IPF(),
    itol = InputTol(max_iter = 50, max_time = 20.0, ϵ_rc = 1.0e-4, ϵ_rb = 1.0e-4, ϵ_pdd = 1.0e-4),
  )
  @test isapprox(stats4.objective, -4.6475314286e02, atol = 1e-1)
  @test stats4.status == :first_order
end

@testset "K2.5 structured LP" begin
  for kmethod in [:gpmr, :trimr]
    for δ_min in [1.0e-2, 0.0]
      stats4 = ripqp(
        QuadraticModel(qps4),
        display = false,
        solve_method = IPF(),
        sp = K2_5StructuredParams(kmethod = kmethod, δ_min = δ_min),
        itol = InputTol(
          max_iter = 50,
          max_time = 20.0,
          ϵ_rc = 1.0e-4,
          ϵ_rb = 1.0e-4,
          ϵ_pdd = 1.0e-4,
        ),
      )
      @test isapprox(stats4.objective, -4.6475314286e02, atol = 1e-1)
      @test stats4.status == :first_order
    end
  end
  stats4 = ripqp(
    QuadraticModel(qps4),
    display = false,
    sp = K2_5StructuredParams(kmethod = :tricg),
    solve_method = IPF(),
    itol = InputTol(max_iter = 50, max_time = 20.0, ϵ_rc = 1.0e-4, ϵ_rb = 1.0e-4, ϵ_pdd = 1.0e-4),
  )
  @test isapprox(stats4.objective, -4.6475314286e02, atol = 1e-1)
  @test stats4.status == :first_order
end
