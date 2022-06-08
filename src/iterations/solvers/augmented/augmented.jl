abstract type AugmentedParams <: SolverParams end

abstract type AugmentedKrylovParams{PT <: AbstractPreconditioner} <: AugmentedParams end

abstract type PreallocatedDataAugmented{T <: Real, S} <: PreallocatedData{T, S} end

abstract type PreallocatedDataAugmentedLDL{T <: Real, S} <: PreallocatedDataAugmented{T, S} end

uses_krylov(pad::PreallocatedDataAugmentedLDL) = false

include("K2LDL.jl")
include("K2_5LDL.jl")
include("K2LDLDense.jl")

abstract type PreallocatedDataAugmentedKrylov{T <: Real, S} <: PreallocatedDataAugmented{T, S} end

uses_krylov(pad::PreallocatedDataAugmentedKrylov) = true

include("K2Krylov.jl")
include("K2_5Krylov.jl")
include("K2Structured.jl")
include("K2_5Structured.jl")
