using RipQP, LLSModels, QPSReader, QuadraticModels, Quadmath
using DelimitedFiles, LinearAlgebra, MatrixMarket, Test

qps1 = readqps("QAFIRO.SIF") #lower bounds
qps2 = readqps("HS21.SIF") # low/upp bounds
qps3 = readqps("HS52.SIF") # free bounds
qps4 = readqps("AFIRO.SIF") # LP
qps5 = readqps("HS35MOD.SIF") # fixed variables

include("solvers/test_augmented.jl")
include("solvers/test_newton.jl")
include("solvers/test_normal.jl")

include("test_multi.jl")
include("test_alternative_methods.jl")
include("test_alternative_problems.jl")
