abstract type NewtonParams <: SolverParams end

abstract type NewtonKrylovParams{PT <: AbstractPreconditioner} <: NewtonParams end

abstract type PreallocatedDataNewton{T <: Real, S} <: PreallocatedData{T, S} end

abstract type PreallocatedDataNewtonKrylov{T <: Real, S} <: PreallocatedDataNewton{T, S} end

include("K3Krylov.jl")
include("K3SKrylov.jl")
include("K3_5Krylov.jl")

abstract type PreallocatedDataNewtonStructured{T <: Real, S} <: PreallocatedDataNewton{T, S} end

# utils for K3_5 gpmr
include("K3_5gpmr_utils.jl")
include("K3SStructured.jl")
include("K3_5Structured.jl")
