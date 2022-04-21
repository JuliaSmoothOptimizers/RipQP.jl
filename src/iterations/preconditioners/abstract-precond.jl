export PreconditionerData
"""
Abstract type that defines preconditioners.
The available preconditioners are:
- Identity
- Jacobi
- Equilibration
- LDLLowPrec32 (K2 only)
"""
abstract type PreconditionerData{T <: Real, S} end

# precond M⁻¹ K N⁻¹
mutable struct LRPrecond{T, S, FM, FMtu, FMaw, FN, FNtu, FNaw}
  M::LinearOperator{T, S, FM, FMtu, FMaw}
  N::LinearOperator{T, S, FN, FNtu, FNaw}
end
