abstract type PreallocatedDataAugmented{T <: Real, S} <: PreallocatedData{T, S} end

abstract type PreallocatedDataAugmentedLDL{T <: Real, S} <: PreallocatedDataAugmented{T, S} end

include("K2LDL.jl")
include("K2_5LDL.jl")

abstract type PreallocatedDataAugmentedKrylov{T <: Real, S} <: PreallocatedDataAugmented{T, S} end

include("K2Krylov.jl")
include("K2_5Krylov.jl")
