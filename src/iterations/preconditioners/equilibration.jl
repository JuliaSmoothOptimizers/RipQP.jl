mutable struct EquilibrationData{T <: Real, S} <: PreconditionerDataK2{T, S}
  P::Diagonal{T, S}
  C_equi::Diagonal{T, S}
end

function Equilibration(
  id::QM_IntData,
  fd::QM_FloatData{T},
  regu::Regularization{T},
  D::AbstractVector{T},
  K::LinearOperator{T},
) where {T <: Real}
  P = Diagonal(similar(D, id.nvar + id.ncon))
  P.diag .= one(T)
  C_equi = Diagonal(similar(D, id.nvar + id.ncon))
  return EquilibrationData(P, C_equi)
end

function update_preconditioner!(
  pdat::EquilibrationData{T},
  pad::PreallocatedData{T},
  itd::IterData{T},
  pt::Point{T},
  id::QM_IntData,
  fd::QM_FloatData{T},
  cnts::Counters,
) where {T <: Real}
  equilibrate_K2!(
    fd.Q.data,
    fd.A,
    pad.D,
    pad.regu.δ,
    id.nvar,
    id.ncon,
    pad.pdat.P,
    pad.pdat.C_equi,
    fd.uplo;
    ϵ = T(1.0e-4),
    max_iter = 100,
  )
  pdat.P.diag .= pdat.P.diag .^ 2
end
