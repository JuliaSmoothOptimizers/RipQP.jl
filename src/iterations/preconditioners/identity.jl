mutable struct IdentityData{T <: Real, S, SI} <: PreconditionerDataK2{T, S}
  P::SI
end

function Identity(
  id::QM_IntData,
  fd::QM_FloatData{T},
  regu::Regularization{T},
  D::AbstractVector{T},
  K::LinearOperator{T},
) where {T <: Real}
  P = I(id.nvar + id.ncon)
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
