@testset "mono_mode" begin
  stats1 = ripqp(QuadraticModel(qps1), iconf = InputConfig(presolve = false))
  @test isapprox(stats1.objective, -1.59078179, atol = 1e-2)
  @test stats1.status == :acceptable

  stats2 = ripqp(QuadraticModel(qps2))
  @test isapprox(stats2.objective, -9.99599999e1, atol = 1e-2)
  @test stats1.status == :acceptable

  stats3 = ripqp(QuadraticModel(qps3))
  @test isapprox(stats3.objective, 5.32664756, atol = 1e-2)
  @test stats3.status == :acceptable
end

@testset "K2_5" begin
  stats1 = ripqp(QuadraticModel(qps1), display = false, iconf = InputConfig(sp = K2_5LDLParams()))
  @test isapprox(stats1.objective, -1.59078179, atol = 1e-2)
  @test stats1.status == :acceptable

  stats2 = ripqp(
    QuadraticModel(qps2),
    display = false,
    iconf = InputConfig(sp = K2_5LDLParams(), mode = :multi),
  )
  @test isapprox(stats2.objective, -9.99599999e1, atol = 1e-2)
  @test stats2.status == :acceptable

  stats3 = ripqp(
    QuadraticModel(qps3),
    display = false,
    iconf = InputConfig(sp = K2_5LDLParams(regul = :dynamic)),
  )
  @test isapprox(stats3.objective, 5.32664756, atol = 1e-2)
  @test stats3.status == :acceptable
end

@testset "KrylovK2" begin
  for precond in [:Identity, :Jacobi, :Equilibration]
    for kmethod in [:minres, :minres_qlp, :symmlq, :dqgmres, :diom]
      stats2 = ripqp(
        QuadraticModel(qps2),
        display = false,
        iconf = InputConfig(
          sp = K2KrylovParams(uplo = :L, kmethod = kmethod, preconditioner = precond),
          history = true,
        ),
        itol = InputTol(
          max_iter = 50,
          max_time = 40.0,
          ϵ_rc = 1.0e-2,
          ϵ_rb = 1.0e-2,
          ϵ_pdd = 1.0e-2,
        ),
      )
      @test isapprox(stats2.objective, -9.99599999e1, atol = 1e-1)
      @test stats2.status == :acceptable

      stats3 = ripqp(
        QuadraticModel(qps3),
        display = false,
        iconf = InputConfig(
          sp = K2KrylovParams(uplo = :U, kmethod = kmethod, preconditioner = precond, form_mat = true),
        ),
        itol = InputTol(
          max_iter = 50,
          max_time = 40.0,
          ϵ_rc = 1.0e-2,
          ϵ_rb = 1.0e-2,
          ϵ_pdd = 1.0e-2,
        ),
      )
      @test isapprox(stats3.objective, 5.32664756, atol = 1e-1)
      @test stats3.status == :acceptable
    end
  end
end

@testset "KrylovK2_5" begin
  for kmethod in [:minres, :minres_qlp, :symmlq]
    stats1 = ripqp(
      QuadraticModel(qps1),
      display = true,
      iconf = InputConfig(
        sp = K2_5KrylovParams(kmethod = kmethod, preconditioner = :Identity),
        history = true,
      ),
      itol = InputTol(max_iter = 50, max_time = 20.0, ϵ_rc = 1.0e-2, ϵ_rb = 1.0e-2, ϵ_pdd = 1.0e-2),
    )
    @test isapprox(stats1.objective, -1.59078179, atol = 1e-1)
    @test stats1.status == :acceptable

    stats2 = ripqp(
      QuadraticModel(qps2),
      display = false,
      iconf = InputConfig(
        sp = K2_5KrylovParams(uplo = :U, kmethod = kmethod, preconditioner = :Jacobi),
        solve_method = :IPF,
      ),
      itol = InputTol(max_iter = 50, max_time = 20.0, ϵ_rc = 1.0e-3, ϵ_rb = 1.0e-3, ϵ_pdd = 1.0e-3),
    )
    @test isapprox(stats2.objective, -9.99599999e1, atol = 1e-1)
    @test stats2.status == :acceptable

    stats3 = ripqp(
      QuadraticModel(qps3),
      display = false,
      iconf = InputConfig(sp = K2_5KrylovParams(kmethod = kmethod, preconditioner = :Jacobi)),
      itol = InputTol(max_iter = 50, max_time = 20.0, ϵ_rc = 1.0e-2, ϵ_rb = 1.0e-2, ϵ_pdd = 1.0e-2),
    )
    @test isapprox(stats3.objective, 5.32664756, atol = 1e-1)
    @test stats3.status == :acceptable
  end
end

@testset "K2 structured LP" begin
  for kmethod in [:trimr, :gpmr]
    for δ_min in [0.0, 1.0e-2]
      stats4 = ripqp(
        QuadraticModel(qps4),
        display = false,
        iconf = InputConfig(
          sp = K2StructuredParams(kmethod = kmethod, δ_min = δ_min),
          solve_method = :IPF,
        ),
      )
      @test isapprox(stats4.objective, -4.6475314286e02, atol = 1e-2)
      @test stats4.status == :acceptable
    end
  end
  stats4 = ripqp(
    QuadraticModel(qps4),
    display = false,
    iconf = InputConfig(sp = K2StructuredParams(kmethod = :tricg), solve_method = :IPF),
  )
  @test isapprox(stats4.objective, -4.6475314286e02, atol = 1e-2)
  @test stats4.status == :acceptable
end

@testset "K2.5 structured LP" begin
  for kmethod in [:trimr, :gpmr]
    for δ_min in [0.0, 1.0e-2]
      stats4 = ripqp(
        QuadraticModel(qps4),
        display = false,
        iconf = InputConfig(
          sp = K2_5StructuredParams(kmethod = kmethod, δ_min = δ_min),
          solve_method = :IPF,
        ),
      )
      @test isapprox(stats4.objective, -4.6475314286e02, atol = 1e-2)
      @test stats4.status == :acceptable
    end
  end
  stats4 = ripqp(
    QuadraticModel(qps4),
    display = false,
    iconf = InputConfig(sp = K2_5StructuredParams(kmethod = :tricg), solve_method = :IPF),
  )
  @test isapprox(stats4.objective, -4.6475314286e02, atol = 1e-2)
  @test stats4.status == :acceptable
end
