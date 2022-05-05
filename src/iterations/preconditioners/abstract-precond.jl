export PreconditionerData, AbstractPreconditioner

abstract type AbstractPreconditioner end

abstract type PreconditionerData{T <: Real, S} end

# precond M⁻¹ K N⁻¹
mutable struct LRPrecond{Op1, Op2}
  M::Op1
  N::Op2
end
