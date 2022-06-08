export PreconditionerData, AbstractPreconditioner

"""
Abstract type for the preconditioners used with a solver using a Krylov method.
"""
abstract type AbstractPreconditioner end

abstract type PreconditionerData{T <: Real, S} end

precond_name(pdat::PreconditionerData) = string(typeof(pdat).name.name)[1:(end - 4)]

# precond M⁻¹ K N⁻¹
mutable struct LRPrecond{Op1, Op2}
  M::Op1
  N::Op2
end
