abstract type NormalParams{T} <: SolverParams{T} end

abstract type NormalKrylovParams{T, PT <: AbstractPreconditioner} <: NormalParams{T} end

abstract type PreallocatedDataNormal{T <: Real, S} <: PreallocatedData{T, S} end

abstract type PreallocatedDataNormalChol{T <: Real, S} <: PreallocatedDataNormal{T, S} end

uses_krylov(pad::PreallocatedDataNormalChol) = false

include("K1CholDense.jl")

abstract type PreallocatedDataNormalKrylov{T <: Real, S} <: PreallocatedDataNormal{T, S} end

uses_krylov(pad::PreallocatedDataNormalKrylov) = true

abstract type PreallocatedDataNormalKrylovStructured{T <: Real, S} <: PreallocatedDataNormal{T, S} end

include("K1Krylov.jl")
include("K1_1Structured.jl")
include("K1_2Structured.jl")
