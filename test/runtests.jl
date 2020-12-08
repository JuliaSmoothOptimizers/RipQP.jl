using RipQP, QPSReader, QuadraticModels
using Test

@testset "mono_mode" begin
    qps1 = readqps("QAFIRO.SIF") #lower bounds
    stats1 = ripqp(QuadraticModel(qps1), mode=:mono)
    @test isapprox(stats1.objective, -1.59078179, atol=1e-2)
    @test stats1.status == :acceptable

    qps2 = readqps("HS21.SIF") # low/upp bounds
    stats2 = ripqp(QuadraticModel(qps2))
    @test isapprox(stats2.objective, -9.99599999e1, atol=1e-2)
    @test stats1.status == :acceptable

    qps3 = readqps("HS52.SIF") # free bounds
    stats3 = ripqp(QuadraticModel(qps3))
    @test isapprox(stats3.objective, 5.32664756, atol=1e-2)
    @test stats3.status == :acceptable
end

@testset "multi_mode" begin
    qps1 = readqps("QAFIRO.SIF") #lower bounds
    stats1 = ripqp(QuadraticModel(qps1), mode=:multi, display=false)
    @test isapprox(stats1.objective, -1.59078179, atol=1e-2)
    @test stats1.status == :acceptable

    qps2 = readqps("HS21.SIF") # low/upp bounds
    stats2 = ripqp(QuadraticModel(qps2), mode=:multi, display=false)
    @test isapprox(stats2.objective, -9.99599999e1, atol=1e-2)
    @test stats2.status == :acceptable

    qps3 = readqps("HS52.SIF") # free bounds
    stats3 = ripqp(QuadraticModel(qps3), mode=:multi, display=false)
    @test isapprox(stats3.objective, 5.32664756, atol=1e-2)
    @test stats3.status == :acceptable
end

@testset "dynamic_regularization" begin
    qps1 = readqps("QAFIRO.SIF") #lower bounds
    stats1 = ripqp(QuadraticModel(qps1), regul=:dynamic, display=false)
    @test isapprox(stats1.objective, -1.59078179, atol=1e-2)
    @test stats1.status == :acceptable

    qps2 = readqps("HS21.SIF") # low/upp bounds
    stats2 = ripqp(QuadraticModel(qps2), regul=:dynamic, display=false)
    @test isapprox(stats2.objective, -9.99599999e1, atol=1e-2)
    @test stats2.status == :acceptable

    qps3 = readqps("HS52.SIF") # free bounds
    stats3 = ripqp(QuadraticModel(qps3), regul=:dynamic, display=false)
    @test isapprox(stats3.objective, 5.32664756, atol=1e-2)
    @test stats3.status == :acceptable
end

@testset "multiple_centrality_corrections_auto" begin
    qps1 = readqps("QAFIRO.SIF") #lower bounds
    stats1 = ripqp(QuadraticModel(qps1), K=-1, display=false)
    @test isapprox(stats1.objective, -1.59078179, atol=1e-2)
    @test stats1.status == :acceptable

    qps2 = readqps("HS21.SIF") # low/upp bounds
    stats2 = ripqp(QuadraticModel(qps2), K=-1, display=false)
    @test isapprox(stats2.objective, -9.99599999e1, atol=1e-2)
    @test stats2.status == :acceptable

    qps3 = readqps("HS52.SIF") # free bounds
    stats3 = ripqp(QuadraticModel(qps3), K=-1, display=false)
    @test isapprox(stats3.objective, 5.32664756, atol=1e-2)
    @test stats3.status == :acceptable
end

@testset "2_centrality_corrections" begin
    qps1 = readqps("QAFIRO.SIF") #lower bounds
    stats1 = ripqp(QuadraticModel(qps1), K=2, display=false)
    @test isapprox(stats1.objective, -1.59078179, atol=1e-2)
    @test stats1.status == :acceptable

    qps2 = readqps("HS21.SIF") # low/upp bounds
    stats2 = ripqp(QuadraticModel(qps2), K=2, display=false)
    @test isapprox(stats2.objective, -9.99599999e1, atol=1e-2)
    @test stats2.status == :acceptable

    qps3 = readqps("HS52.SIF") # free bounds
    stats3 = ripqp(QuadraticModel(qps3), K=2, display=false)
    @test isapprox(stats3.objective, 5.32664756, atol=1e-2)
    @test stats3.status == :acceptable
end
