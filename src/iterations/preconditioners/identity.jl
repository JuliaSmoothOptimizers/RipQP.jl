export Identity

"""
    preconditioner = Identity()

Tells RipQP not to use a preconditioner.
"""
mutable struct Identity <: AbstractPreconditioner
end

mutable struct IdentityData{T <: Real, S, SI <: UniformScaling} <: PreconditionerData{T, S}
  P::SI
end

function PreconditionerData(
  sp::AugmentedKrylovParams{Identity},
  id::QM_IntData,
  fd::QM_FloatData{T},
  regu::Regularization{T},
  D::AbstractVector{T},
  K::Union{LinearOperator{T}, AbstractMatrix{T}},
) where {T <: Real}
  P = I
  return IdentityData{T, typeof(fd.c), typeof(P)}(P)
end

function PreconditionerData(
  sp::NewtonKrylovParams{Identity},
  id::QM_IntData,
  fd::QM_FloatData{T},
  regu::Regularization{T},
  K::Union{LinearOperator{T}, AbstractMatrix{T}},
) where {T <: Real, PT <: Identity}
  P = I
  return IdentityData{T, typeof(fd.c), typeof(P)}(P)
end

function update_preconditioner!(
  pdat::IdentityData{T},
  pad::PreallocatedData{T},
  itd::IterData{T},
  pt::Point{T},
  id::QM_IntData,
  fd::QM_FloatData{T},
  cnts::Counters,
) where {T <: Real} end
