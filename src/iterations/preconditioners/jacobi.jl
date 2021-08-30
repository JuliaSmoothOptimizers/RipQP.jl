mutable struct JacobiData{T <: Real, S, L <: LinearOperator} <: PreconditionerDataK2{T, S}
  P::L
  diagQ::S
  invDiagK::S
end

function Jacobi(
  id::QM_IntData,
  fd::QM_FloatData{T},
  regu::Regularization{T},
  D::AbstractVector{T},
  K::LinearOperator{T},
) where {T <: Real}
  invDiagK = (one(T) / regu.δ) .* fill!(similar(fd.c, id.nvar + id.ncon), one(T))
  diagQ = get_diag_Q_dense(fd.Q)
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
  if typeof(pad) <: PreallocatedData_K2_5Krylov
    pad.pdat.invDiagK[1:(id.nvar)] .=
      abs.(one(T) ./ (pad.D .- (pad.pdat.diagQ .* pad.sqrtX1X2 .^ 2)))
  else
    pad.pdat.invDiagK[1:(id.nvar)] .= abs.(one(T) ./ (pad.D .- pad.pdat.diagQ))
  end
  pad.pdat.invDiagK[(id.nvar + 1):end] .= one(T) / pad.regu.δ
end
