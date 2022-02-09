
@testset "Krylov K1" begin
  for kmethod in [:cg, :cg_lanczos, :cr]
    stats4 = ripqp(
      QuadraticModel(qps4),
      display = false,
      iconf = InputConfig(
        sp = K1KrylovParams(kmethod = kmethod),
        solve_method = :IPF,
        history = true,
      ),
      itol = InputTol(max_iter = 50, max_time = 20.0, ϵ_rc = 1.0e-4, ϵ_rb = 1.0e-4, ϵ_pdd = 1.0e-4),
    )
    @test isapprox(stats4.objective, -4.6475314286e02, atol = 1e-2)
    @test stats4.status == :acceptable
  end

  stats4 = ripqp(
    QuadraticModel(qps4),
    display = false,
    iconf = InputConfig(sp = K1KrylovParams(uplo = :U), solve_method = :PC, history = true),
    itol = InputTol(max_iter = 50, max_time = 20.0, ϵ_rc = 1.0e-4, ϵ_rb = 1.0e-4, ϵ_pdd = 1.0e-4),
  )
  @test isapprox(stats4.objective, -4.6475314286e02, atol = 1e-2)
  @test stats4.status == :acceptable
end
