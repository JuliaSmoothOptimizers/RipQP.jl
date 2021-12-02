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

@testset "multi_mode" begin
  stats1 = ripqp(QuadraticModel(qps1), iconf = InputConfig(mode = :multi), display = false)
  @test isapprox(stats1.objective, -1.59078179, atol = 1e-2)
  @test stats1.status == :acceptable

  stats2 = ripqp(QuadraticModel(qps2), iconf = InputConfig(mode = :multi), display = false)
  @test isapprox(stats2.objective, -9.99599999e1, atol = 1e-2)
  @test stats2.status == :acceptable

  stats3 = ripqp(QuadraticModel(qps3), iconf = InputConfig(mode = :multi), display = false)
  @test isapprox(stats3.objective, 5.32664756, atol = 1e-2)
  @test stats3.status == :acceptable
end

@testset "Float128" begin
  qm128_1 = createQuadraticModel128(qps1)
  stats1 = ripqp(
    qm128_1,
    itol = InputTol(ϵ_rb32 = 0.1, ϵ_rb64 = 0.01),
    iconf = InputConfig(mode = :multi, normalize_rtol = false),
    display = false,
  )
  @test isapprox(stats1.objective, -1.59078179, atol = 1e-2)
  @test stats1.status == :acceptable

  qm128_2 = createQuadraticModel128(qps2)
  stats2 = ripqp(qm128_2, display = false)
  @test isapprox(stats2.objective, -9.99599999e1, atol = 1e-2)
  @test stats2.status == :acceptable
end
