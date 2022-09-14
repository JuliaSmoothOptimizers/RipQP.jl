abstract type AugmentedParams{T} <: SolverParams{T} end

abstract type AugmentedKrylovParams{T, PT <: AbstractPreconditioner} <: AugmentedParams{T} end

abstract type PreallocatedDataAugmented{T <: Real, S} <: PreallocatedData{T, S} end

abstract type PreallocatedDataAugmentedLDL{T <: Real, S} <: PreallocatedDataAugmented{T, S} end

uses_krylov(pad::PreallocatedDataAugmentedLDL) = false

include("K2LDL.jl")
include("K2_5LDL.jl")
include("K2LDLDense.jl")

abstract type PreallocatedDataAugmentedKrylov{T <: Real, S} <: PreallocatedDataAugmented{T, S} end

uses_krylov(pad::PreallocatedDataAugmentedKrylov) = true

abstract type PreallocatedDataAugmentedKrylovStructured{T <: Real, S} <:
              PreallocatedDataAugmented{T, S} end

include("K2Krylov.jl")
include("K2_5Krylov.jl")
include("K2Structured.jl")
include("K2_5Structured.jl")
