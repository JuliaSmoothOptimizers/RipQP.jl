using RipQP, QPSReader, QuadraticModels, Quadmath
using Test

function createQuadraticModel128(qpdata; name="qp_pb")
    return QuadraticModel(convert(Array{Float128}, qpdata.c), qpdata.qrows, qpdata.qcols,
            convert(Array{Float128}, qpdata.qvals),
            Arows=qpdata.arows, Acols=qpdata.acols,
            Avals=convert(Array{Float128}, qpdata.avals),
            lcon=convert(Array{Float128}, qpdata.lcon),
            ucon=convert(Array{Float128}, qpdata.ucon),
            lvar=convert(Array{Float128}, qpdata.lvar),
            uvar=convert(Array{Float128}, qpdata.uvar),
            c0=Float128(qpdata.c0), x0 = zeros(Float128, length(qpdata.c)), name=name)
end

@testset "mono_mode" begin
    qps1 = readqps("QAFIRO.SIF") #lower bounds
    stats1 = ripqp(QuadraticModel(qps1))
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
    stats1 = ripqp(QuadraticModel(qps1), iconf = input_config(mode = :multi), display=false)
    @test isapprox(stats1.objective, -1.59078179, atol=1e-2)
    @test stats1.status == :acceptable

    qps2 = readqps("HS21.SIF") # low/upp bounds
    stats2 = ripqp(QuadraticModel(qps2), iconf = input_config(mode = :multi), display=false)
    @test isapprox(stats2.objective, -9.99599999e1, atol=1e-2)
    @test stats2.status == :acceptable

    qps3 = readqps("HS52.SIF") # free bounds
    stats3 = ripqp(QuadraticModel(qps3), iconf = input_config(mode = :multi), display=false)
    @test isapprox(stats3.objective, 5.32664756, atol=1e-2)
    @test stats3.status == :acceptable
end

@testset "dynamic_regularization" begin
    qps1 = readqps("QAFIRO.SIF") #lower bounds
    stats1 = ripqp(QuadraticModel(qps1), iconf = input_config(regul=:dynamic), display=false)
    @test isapprox(stats1.objective, -1.59078179, atol=1e-2)
    @test stats1.status == :acceptable

    qps2 = readqps("HS21.SIF") # low/upp bounds
    stats2 = ripqp(QuadraticModel(qps2), iconf = input_config(regul=:dynamic), display=false)
    @test isapprox(stats2.objective, -9.99599999e1, atol=1e-2)
    @test stats2.status == :acceptable

    qps3 = readqps("HS52.SIF") # free bounds
    stats3 = ripqp(QuadraticModel(qps3), iconf = input_config(regul=:dynamic), display=false)
    @test isapprox(stats3.objective, 5.32664756, atol=1e-2)
    @test stats3.status == :acceptable
end

@testset "centrality_corrections" begin
    qps1 = readqps("QAFIRO.SIF") #lower bounds
    stats1 = ripqp(QuadraticModel(qps1), iconf = input_config(K=-1), display=false) # automatic centrality corrections computation
    @test isapprox(stats1.objective, -1.59078179, atol=1e-2)
    @test stats1.status == :acceptable

    qps2 = readqps("HS21.SIF") # low/upp bounds
    stats2 = ripqp(QuadraticModel(qps2), iconf = input_config(K=2), display=false)
    @test isapprox(stats2.objective, -9.99599999e1, atol=1e-2)
    @test stats2.status == :acceptable

    qps3 = readqps("HS52.SIF") # free bounds
    stats3 = ripqp(QuadraticModel(qps3), iconf = input_config(K=2), display=false)
    @test isapprox(stats3.objective, 5.32664756, atol=1e-2)
    @test stats3.status == :acceptable
end

@testset "Float128" begin
    qps1 = readqps("QAFIRO.SIF") #lower bounds
    qm128_1 = createQuadraticModel128(qps1)
    stats1 = ripqp(qm128_1, itol = input_tol(ϵ_rb32=0.1, ϵ_rb64=0.01), iconf = input_config(mode=:multi, normalize_rtol=false), display=false)
    @test isapprox(stats1.objective, -1.59078179, atol=1e-2)
    @test stats1.status == :acceptable

    qps2 = readqps("HS21.SIF") # low/upp bounds
    qm128_2 = createQuadraticModel128(qps2)
    stats2 = ripqp(qm128_2, display=false)
    @test isapprox(stats2.objective, -9.99599999e1, atol=1e-2)
    @test stats2.status == :acceptable
end

@testset "refinement" begin
    qps1 = readqps("QAFIRO.SIF") #lower bounds
    stats1 = ripqp(QuadraticModel(qps1), iconf = input_config(refinement=:zoom), display=false) # automatic centrality corrections computation
    @test isapprox(stats1.objective, -1.59078179, atol=1e-2)
    @test stats1.status == :acceptable

    qps1 = readqps("QAFIRO.SIF") #lower bounds
    stats1 = ripqp(QuadraticModel(qps1), iconf = input_config(refinement=:ref), display=false) # automatic centrality corrections computation
    @test isapprox(stats1.objective, -1.59078179, atol=1e-2)
    @test stats1.status == :acceptable

    qps2 = readqps("HS21.SIF") # low/upp bounds
    stats2 = ripqp(QuadraticModel(qps2), iconf = input_config(mode=:multi, refinement=:multizoom),  display=false)
    @test isapprox(stats2.objective, -9.99599999e1, atol=1e-2)
    @test stats2.status == :acceptable

    qps3 = readqps("HS52.SIF") # free bounds
    stats3 = ripqp(QuadraticModel(qps3), iconf = input_config(mode=:multi, refinement=:multiref),  display=false)
    @test isapprox(stats3.objective, 5.32664756, atol=1e-2)
    @test stats3.status == :acceptable
end

@testset "K2_5" begin
    qps1 = readqps("QAFIRO.SIF") #lower bounds
    stats1 = ripqp(QuadraticModel(qps1), display=false, iconf = input_config(solve! = solve_K2_5!))
    @test isapprox(stats1.objective, -1.59078179, atol=1e-2)
    @test stats1.status == :acceptable

    qps2 = readqps("HS21.SIF") # low/upp bounds
    stats2 = ripqp(QuadraticModel(qps2), display=false, iconf = input_config(solve! = solve_K2_5!, mode = :multi))
    @test isapprox(stats2.objective, -9.99599999e1, atol=1e-2)
    @test stats2.status == :acceptable

    qps3 = readqps("HS52.SIF") # free bounds
    stats3 = ripqp(QuadraticModel(qps3), display=false, iconf = input_config(solve! = solve_K2_5!, regul = :dynamic))
    @test isapprox(stats3.objective, 5.32664756, atol=1e-2)
    @test stats3.status == :acceptable
end
