export PreconditionerData
"""
Abstract type that defines preconditioners.
The available preconditioners are:
- Identity
- Jacobi
- Equilibration
"""
abstract type PreconditionerData{T <: Real, S} end

include("identity.jl")
include("jacobi.jl")
include("equilibration.jl")
