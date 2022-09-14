export Equilibration

"""
    preconditioner = Equilibration()

Preconditioner using the equilibration algorithm in infinity norm.
Works with:
- [`K2KrylovParams`](@ref)
- [`K3SKrylovParams`](@ref)
- [`K3_5KrylovParams`](@ref)
"""
mutable struct Equilibration <: AbstractPreconditioner end

mutable struct EquilibrationData{T <: Real, S} <: PreconditionerData{T, S}
  P::Diagonal{T, S}
  C_equi::Diagonal{T, S}
end

function PreconditionerData(
  sp::AugmentedKrylovParams{T, Equilibration},
  id::QM_IntData,
  fd::QM_FloatData{T},
  regu::Regularization{T},
  D::AbstractVector{T},
  K::Union{LinearOperator{T}, AbstractMatrix{T}},
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

mutable struct EquilibrationK3SData{T <: Real, S, L <: LinearOperator{T}} <:
               PreconditionerData{T, S}
  P::L
  d_l::S
  d_u::S
end

function PreconditionerData(
  sp::NewtonKrylovParams{T, Equilibration},
  id::QM_IntData,
  fd::QM_FloatData{T, S},
  regu::Regularization{T},
  K::Union{LinearOperator{T}, AbstractMatrix{T}},
) where {T <: Real, S}
  d_l = fill!(S(undef, id.nlow), zero(T))
  d_u = fill!(S(undef, id.nupp), zero(T))
  P = BlockDiagonalOperator(opEye(T, id.nvar + id.ncon), opDiagonal(d_l), opDiagonal(d_u))
  return EquilibrationK3SData(P, d_l, d_u)
end

function update_preconditioner!(
  pdat::EquilibrationK3SData{T},
  pad::PreallocatedDataK3SKrylov{T},
  itd::IterData{T},
  pt::Point{T},
  id::QM_IntData,
  fd::QM_FloatData{T},
  cnts::Counters,
) where {T <: Real}
  TS = typeof(pad.KS)
  if TS <: GmresSolver || TS <: DqgmresSolver
    pad.pdat.d_l .= sqrt.(one(T) ./ max.(one(T), pad.x_m_lvar_div_s_l))
    pad.pdat.d_u .= sqrt.(one(T) ./ max.(one(T), pad.uvar_m_x_div_s_u))
  else
    pad.pdat.d_l .= one(T) ./ max.(one(T), pad.x_m_lvar_div_s_l)
    pad.pdat.d_u .= one(T) ./ max.(one(T), pad.uvar_m_x_div_s_u)
  end
end

function update_preconditioner!(
  pdat::EquilibrationK3SData{T},
  pad::PreallocatedDataK3_5Krylov{T},
  itd::IterData{T},
  pt::Point{T},
  id::QM_IntData,
  fd::QM_FloatData{T},
  cnts::Counters,
) where {T <: Real}
  TS = typeof(pad.KS)
  if TS <: GmresSolver || TS <: DqgmresSolver
    pad.pdat.d_l .= sqrt.(one(T) ./ max.(pt.s_l, itd.x_m_lvar))
    pad.pdat.d_u .= sqrt.(one(T) ./ max.(pt.s_u, itd.uvar_m_x))
  else
    pad.pdat.d_l .= one(T) ./ max.(pt.s_l, itd.x_m_lvar)
    pad.pdat.d_u .= one(T) ./ max.(pt.s_u, itd.uvar_m_x)
  end
end
