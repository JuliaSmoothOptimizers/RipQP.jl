abstract type PreallocatedDataAugmented{T <: Real, S} <: PreallocatedData{T, S} end

abstract type PreallocatedDataAugmentedLDL{T <: Real, S} <: PreallocatedDataAugmented{T, S} end

include("K2LDL.jl")
include("K2_5LDL.jl")
include("K2LDLDense.jl")

abstract type PreallocatedDataAugmentedKrylov{T <: Real, S} <: PreallocatedDataAugmented{T, S} end

include("K2Krylov.jl")
include("K2_5Krylov.jl")

abstract type PreallocatedDataAugmentedStructured{T <: Real, S} <: PreallocatedDataAugmented{T, S} end

include("K2Structured.jl")
include("K2_5Structured.jl")
