function createQuadraticModelT(qpdata; T = Float128, name = "qp_pb")
  return QuadraticModel(
    convert(Array{T}, qpdata.c),
    qpdata.qrows,
    qpdata.qcols,
    convert(Array{T}, qpdata.qvals),
    Arows = qpdata.arows,
    Acols = qpdata.acols,
    Avals = convert(Array{T}, qpdata.avals),
    lcon = convert(Array{T}, qpdata.lcon),
    ucon = convert(Array{T}, qpdata.ucon),
    lvar = convert(Array{T}, qpdata.lvar),
    uvar = convert(Array{T}, qpdata.uvar),
    c0 = T(qpdata.c0),
    x0 = zeros(T, length(qpdata.c)),
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

@testset "Float16, Float32, Float128" begin
  qm128_1 = createQuadraticModelT(qps1, T = Float128)
  stats1 = ripqp(
    qm128_1,
    itol = InputTol(ϵ_rb32 = 0.1, ϵ_rb64 = 0.01),
    iconf = InputConfig(mode = :multi, normalize_rtol = false),
    display = false,
  )
  @test isapprox(stats1.objective, -1.59078179, atol = 1e-2)
  @test stats1.status == :acceptable

  for T ∈ [Float32, Float128]
    qmT_2 = createQuadraticModelT(qps2, T = T)
    stats2 = ripqp(qmT_2, display = false)
    @test isapprox(stats2.objective, -9.99599999e1, atol = 1e-2)
    @test stats2.status == :acceptable
  end

  qm16 = QuadraticModel(
    Float16.(c),
    Float16.(Q),
    A = Float16.(A),
    lcon = Float16.(b),
    ucon = Float16.(b),
    lvar = Float16.(l),
    uvar = Float16.(u),
    c0 = Float16(0.0),
    x0 = zeros(Float16, 3),
    name = "QM16",
  )
  stats_dense = ripqp(qm16, itol = InputTol(ϵ_pdd = 1.0e-1, ϵ_rb = 1.0e-1, ϵ_rc = 1.0e-1))
  @test isapprox(stats_dense.objective, 1.1249999990782493, atol = 1e-2)
  @test stats_dense.status == :acceptable
end
