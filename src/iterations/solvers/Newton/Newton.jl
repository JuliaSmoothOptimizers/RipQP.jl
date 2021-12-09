abstract type PreallocatedDataNewton{T <: Real, S} <: PreallocatedData{T, S} end

abstract type PreallocatedDataNewtonKrylov{T <: Real, S} <: PreallocatedDataNewton{T, S} end

include("K3Krylov.jl")
include("K3_5Krylov.jl")

abstract type PreallocatedDataNewtonStructured{T <: Real, S} <: PreallocatedDataNewton{T, S} end

include("K3_5Structured.jl")
