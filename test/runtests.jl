using HSL, LinearOperators, LLSModels, NLPModelsModifiers, QPSReader, QuadraticModels
using DelimitedFiles, DoubleFloats, LinearAlgebra, MatrixMarket, SparseArrays, Test
using SuiteSparse  # test cholmod
using RipQP

include("fixtures.jl")

include("solvers/test_augmented.jl")
include("solvers/test_newton.jl")
include("solvers/test_normal.jl")

include("test_multi.jl")
include("test_alternative_methods.jl")
include("test_alternative_problems.jl")
