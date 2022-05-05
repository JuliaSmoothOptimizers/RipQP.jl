export Jacobi

"""
    preconditioner = Jacobi()

Preconditioner using the inverse of the diagonal of the system to solve.
Works with:
- [`K2KrylovParams`](@ref)
- [`K2_5KrylovParams`](@ref)
"""
mutable struct Jacobi <: AbstractPreconditioner end

mutable struct JacobiData{T <: Real, S, L <: LinearOperator} <: PreconditionerData{T, S}
  P::L
  diagQ::S
  invDiagK::S
end

function PreconditionerData(
  sp::AugmentedKrylovParams{Jacobi},
  id::QM_IntData,
  fd::QM_FloatData{T},
  regu::Regularization{T},
  D::AbstractVector{T},
  K::Union{LinearOperator{T}, AbstractMatrix{T}},
) where {T <: Real}
  invDiagK = (one(T) / regu.δ) .* fill!(similar(fd.c, id.nvar + id.ncon), one(T))
  diagQ = get_diag_Q_dense(fd.Q, fd.uplo)
  invDiagK[1:(id.nvar)] .= .-one(T) ./ (D .- diagQ)
  P = opDiagonal(invDiagK)
  return JacobiData{eltype(diagQ), typeof(diagQ), typeof(P)}(P, diagQ, invDiagK)
end

function update_preconditioner!(
  pdat::JacobiData{T},
  pad::PreallocatedData{T},
  itd::IterData{T},
  pt::Point{T},
  id::QM_IntData,
  fd::QM_FloatData{T},
  cnts::Counters,
) where {T <: Real}
  if typeof(pad) <: PreallocatedDataK2_5Krylov
    pad.pdat.invDiagK[1:(id.nvar)] .=
      abs.(one(T) ./ (pad.D .- (pad.pdat.diagQ .* pad.sqrtX1X2 .^ 2)))
  else
    pad.pdat.invDiagK[1:(id.nvar)] .= abs.(one(T) ./ (pad.D .- pad.pdat.diagQ))
  end
  pad.pdat.invDiagK[(id.nvar + 1):end] .= one(T) / pad.regu.δ
end
