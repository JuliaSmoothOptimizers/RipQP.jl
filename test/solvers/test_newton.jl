@testset "Krylov K3" begin
  for kmethod in [:bilq, :bicgstab, :usymlq, :usymqr, :qmr, :diom, :fom, :gmres, :dqgmres]
    stats2 = ripqp(
      QuadraticModel(qps2),
      display = false,
      sp = K3KrylovParams(kmethod = kmethod),
      solve_method = IPF(),
      history = true,
      itol = InputTol(max_iter = 50, max_time = 20.0, ϵ_rc = 1.0e-4, ϵ_rb = 1.0e-4, ϵ_pdd = 1.0e-4),
    )
    @test isapprox(stats2.objective, -9.99599999e1, atol = 1e-1)
    @test stats2.status == :first_order
  end
  stats3 = ripqp(
    QuadraticModel(qps3),
    display = false,
    sp = K3KrylovParams(uplo = :U, kmethod = :gmres),
    itol = InputTol(max_iter = 50, max_time = 20.0, ϵ_rc = 1.0e-4, ϵ_rb = 1.0e-4, ϵ_pdd = 1.0e-4),
  )
  @test isapprox(stats3.objective, 5.32664756, atol = 1e-1)
  @test stats3.status == :first_order
end

@testset "KrylovK3_5" begin
  for kmethod in [:minres, :minres_qlp]
    stats1 = ripqp(
      QuadraticModel(qps1),
      display = false,
      sp = K3_5KrylovParams(kmethod = kmethod),
      history = true,
      itol = InputTol(max_iter = 50, max_time = 20.0, ϵ_rc = 1.0e-2, ϵ_rb = 1.0e-2, ϵ_pdd = 1.0e-2),
    )
    @test isapprox(stats1.objective, -1.59078179, atol = 1e-1)
    @test stats1.status == :first_order

    stats2 = ripqp(
      QuadraticModel(qps2),
      display = false,
      sp = K3_5KrylovParams(uplo = :U, kmethod = kmethod),
      solve_method = IPF(),
      itol = InputTol(max_iter = 50, max_time = 20.0, ϵ_rc = 1.0e-3, ϵ_rb = 1.0e-3, ϵ_pdd = 1.0e-3),
    )
    @test isapprox(stats2.objective, -9.99599999e1, atol = 1e-1)
    @test stats2.status == :first_order
  end

  for kmethod in [:minres, :gmres]
    stats2 = ripqp(
      QuadraticModel(qps2),
      display = false,
      sp = K3_5KrylovParams(uplo = :U, kmethod = kmethod, preconditioner = Equilibration()),
      solve_method = IPF(),
      itol = InputTol(max_iter = 50, max_time = 20.0, ϵ_rc = 1.0e-2, ϵ_rb = 1.0e-2, ϵ_pdd = 1.0e-2),
    )
    @test isapprox(stats2.objective, -9.99599999e1, atol = 1e0)
    @test stats2.status == :first_order

    stats3 = ripqp(
      QuadraticModel(qps3),
      display = false,
      sp = K3_5KrylovParams(kmethod = kmethod, preconditioner = Equilibration()),
      itol = InputTol(max_iter = 50, max_time = 20.0, ϵ_rc = 1.0e-2, ϵ_rb = 1.0e-2, ϵ_pdd = 1.0e-2),
    )
    @test isapprox(stats3.objective, 5.32664756, atol = 1e-1)
    @test stats3.status == :first_order
  end
end

@testset "KrylovK3S" begin
  stats2 = ripqp(
    QuadraticModel(qps2),
    display = true,
    sp = K3SKrylovParams(uplo = :U, kmethod = :minres_qlp),
    solve_method = IPF(),
    itol = InputTol(max_iter = 50, max_time = 20.0, ϵ_rc = 1.0e-2, ϵ_rb = 1.0e-2, ϵ_pdd = 1.0e-2),
  )
  @test isapprox(stats2.objective, -9.99599999e1, atol = 1e0)
  @test stats2.status == :first_order

  stats3 = ripqp(
    QuadraticModel(qps3),
    display = false,
    sp = K3SKrylovParams(kmethod = :minres_qlp),
    itol = InputTol(max_iter = 50, max_time = 20.0, ϵ_rc = 1.0e-2, ϵ_rb = 1.0e-2, ϵ_pdd = 1.0e-2),
  )
  @test isapprox(stats3.objective, 5.32664756, atol = 1e-1)
  @test stats3.status == :first_order

  stats2 = ripqp(
    QuadraticModel(qps2),
    display = false,
    sp = K3SKrylovParams(uplo = :U, kmethod = :minres, preconditioner = Equilibration()),
    solve_method = IPF(),
    itol = InputTol(max_iter = 50, max_time = 20.0, ϵ_rc = 1.0e-2, ϵ_rb = 1.0e-2, ϵ_pdd = 1.0e-2),
  )
  @test isapprox(stats2.objective, -9.99599999e1, atol = 1e0)
  @test stats2.status == :first_order

  stats3 = ripqp(
    QuadraticModel(qps3),
    display = false,
    sp = K3SKrylovParams(kmethod = :minres, preconditioner = Equilibration()),
    itol = InputTol(max_iter = 50, max_time = 20.0, ϵ_rc = 1.0e-2, ϵ_rb = 1.0e-2, ϵ_pdd = 1.0e-2),
  )
  @test isapprox(stats3.objective, 5.32664756, atol = 1e-1)
  @test stats3.status == :first_order
end

@testset "K3_5 structured" begin
  for kmethod in [:gpmr, :trimr]
    stats3 = ripqp(
      QuadraticModel(qps3),
      display = false,
      sp = K3_5StructuredParams(kmethod = :gpmr),
      solve_method = IPF(),
      history = true,
      itol = InputTol(
        max_iter = 100,
        max_time = 20.0,
        ϵ_rc = 1.0e-2,
        ϵ_rb = 1.0e-2,
        ϵ_pdd = 1.0e-2,
      ),
    )
    @test isapprox(stats3.objective, 5.32664756, atol = 1e-1)
    @test stats3.status == :first_order
  end
end

@testset "K3S structured" begin
  for kmethod in [:gpmr, :trimr]
    stats3 = ripqp(
      QuadraticModel(qps3),
      display = false,
      sp = K3SStructuredParams(kmethod = kmethod),
      solve_method = IPF(),
      history = true,
      itol = InputTol(
        max_iter = 100,
        max_time = 20.0,
        ϵ_rc = 1.0e-2,
        ϵ_rb = 1.0e-2,
        ϵ_pdd = 1.0e-2,
      ),
    )
    @test isapprox(stats3.objective, 5.32664756, atol = 1e-1)
    @test stats3.status == :first_order
  end
end
