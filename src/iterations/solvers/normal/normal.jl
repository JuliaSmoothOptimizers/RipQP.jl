abstract type NormalParams <: SolverParams end

abstract type PreallocatedDataNormal{T <: Real, S} <: PreallocatedData{T, S} end

abstract type PreallocatedDataNormalKrylov{T <: Real, S} <: PreallocatedDataNormal{T, S} end

include("K1Krylov.jl")

abstract type PreallocatedDataNormalStructured{T <: Real, S} <: PreallocatedDataNormal{T, S} end

include("K1_1Structured.jl")
include("K1_2Structured.jl")
