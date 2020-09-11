using RipQP, QPSReader, QuadraticModels
using Test

@testset "RipQP.jl" begin
    qps1 = readqps("QAFIRO.SIF") #lower bounds
    stats1 = ripQP(QuadraticModel(qps1))
    @test isapprox(stats1.objective, -1.59078179, atol=1e-2)
end
