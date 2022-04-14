mutable struct IdentityData{T <: Real, S, SI <: UniformScaling} <: PreconditionerData{T, S}
  P::SI
end

function Identity(
  id::QM_IntData,
  fd::QM_FloatData{T},
  regu::Regularization{T},
  D::AbstractVector{T},
  K::Union{LinearOperator{T}, AbstractMatrix{T}},
) where {T <: Real}
  P = I
  return IdentityData{T, typeof(fd.c), typeof(P)}(P)
end

function Identity(
  id::QM_IntData,
  fd::QM_FloatData{T},
  regu::Regularization{T},
  K::Union{LinearOperator{T}, AbstractMatrix{T}},
) where {T <: Real}
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
