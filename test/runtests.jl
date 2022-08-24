using LinearOperators, LLSModels, NLPModelsModifiers, QPSReader, QuadraticModels, Quadmath
using DelimitedFiles, LinearAlgebra, MatrixMarket, SparseArrays, Test
using SuiteSparse # test cholmod
using RipQP

qps1 = readqps("QAFIRO.SIF") #lower bounds
qps2 = readqps("HS21.SIF") # low/upp bounds
qps3 = readqps("HS52.SIF") # free bounds
qps4 = readqps("AFIRO.SIF") # LP
qps5 = readqps("HS35MOD.SIF") # fixed variables

Q = [
  6.0 2.0 1.0
  2.0 5.0 2.0
  1.0 2.0 4.0
]
c = [-8.0; -3; -3]
A = [
  1.0 0.0 1.0
  0.0 2.0 1.0
]
b = [0.0; 3]
l = [0.0; 0; 0]
u = [Inf; Inf; Inf]

include("solvers/test_augmented.jl")
include("solvers/test_newton.jl")
include("solvers/test_normal.jl")

include("test_multi.jl")
include("test_alternative_methods.jl")
include("test_alternative_problems.jl")
