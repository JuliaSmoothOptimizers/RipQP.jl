using RipQP, QPSReader, QuadraticModels
using Test

@testset "RipQP.jl" begin
    qps1 = readqps("QAFIRO.SIF") #lower bounds
    stats1 = ripqp(QuadraticModel(qps1), mode=:mono)
    @test isapprox(stats1.objective, -1.59078179, atol=1e-2)
    @test stats1.status == :acceptable
    stats1m = ripqp(QuadraticModel(qps1), mode=:multi)
    @test isapprox(stats1m.objective, -1.59078179, atol=1e-2)
    @test stats1m.status == :acceptable

    qps2 = readqps("HS21.SIF") # low/upp bounds
    stats2 = ripqp(QuadraticModel(qps2))
    @test isapprox(stats2.objective, -9.99599999e1, atol=1e-2)
    @test stats1.status == :acceptable
    stats2m = ripqp(QuadraticModel(qps2), mode=:multi)
    @test isapprox(stats2m.objective, -9.99599999e1, atol=1e-2)
    @test stats2m.status == :acceptable

    qps3 = readqps("HS52.SIF") # free bounds
    stats3 = ripqp(QuadraticModel(qps3))
    @test isapprox(stats3.objective, 5.32664756, atol=1e-2)
    @test stats3.status == :acceptable
    stats3m = ripqp(QuadraticModel(qps3), mode=:multi)
    @test isapprox(stats3m.objective, 5.32664756, atol=1e-2)
    @test stats3m.status == :acceptable
end
