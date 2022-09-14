export LLDL

"""
    preconditioner = LLDL(; T = Float64, mem = 0, droptol = 0.0)

Preconditioner for [`K2KrylovParams`](@ref) using a limited LDL factorization in precision `T`.
See [`LimitedLDLFactorizations.jl`](https://github.com/JuliaSmoothOptimizers/LimitedLDLFactorizations.jl) for the `mem` and `droptol` parameters.
"""
mutable struct LLDL{FloatType <: DataType, Tv} <: AbstractPreconditioner
  T::FloatType
  mem::Int
  droptol::Tv
end

LLDL(; T::DataType = Float64, mem::Int = 0, droptol = 0.0) = LLDL(T, mem, T(droptol))

mutable struct LLDLStor{F, T}
  Fact::F
  mem::Int
  droptol::T
end

mutable struct LLDLData{T <: Real, S, Tlow, Op <: Union{LinearOperator, LRPrecond}, F} <:
               PreconditionerData{T, S}
  K::Symmetric{T, SparseMatrixCSC{T, Int}}
  regu::Regularization{Tlow}
  LLDLS::LLDLStor{F, T}
  fact_fail::Bool # true if factorization failed
  P::Op
end

ldivmem!(res, LLDLS::LLDLStor, x) = ldiv!(res, LLDLS.Fact, x)

function PreconditionerData(
  sp::AugmentedKrylovParams{T, <:LLDL},
  id::QM_IntData,
  fd::QM_FloatData{T},
  regu::Regularization{T},
  D::AbstractVector{T},
  K,
) where {T <: Real}
  Tlow = sp.preconditioner.T
  @assert fd.uplo == :L
  @assert sp.form_mat = true
  regu_precond = Regularization(
    -Tlow(D[1]),
    Tlow(max(regu.δ, sqrt(eps(Tlow)))),
    sqrt(eps(Tlow)),
    sqrt(eps(Tlow)),
    :classic,
  )
  K_fact = lldl(K.data, memory = sp.preconditioner.mem)
  K_fact.D .= abs.(K_fact.D)
  LLDLS = LLDLStor(K_fact, sp.preconditioner.mem, sp.preconditioner.droptol)
  P = LinearOperator(
    Tlow,
    id.nvar + id.ncon,
    id.nvar + id.ncon,
    true,
    true,
    (res, v) -> ldivmem!(res, LLDLS, v),
  )

  return LLDLData{T, Vector{T}, Tlow, typeof(P), typeof(K_fact)}(K, regu_precond, LLDLS, false, P)
end

function update_preconditioner!(
  pdat::LLDLData{T},
  pad::PreallocatedData{T},
  itd::IterData{T},
  pt::Point{T},
  id::QM_IntData,
  fd::Abstract_QM_FloatData{T},
  cnts::Counters,
) where {T <: Real}
  if itd.μ ≤ sqrt(eps(T))
    pdat.LLDLS.Fact =
      lldl(pad.K.data, memory = pdat.LLDLS.mem, droptol = pdat.LLDLS.droptol * itd.μ)
  else
    pdat.LLDLS.Fact = lldl(pad.K.data, memory = pdat.LLDLS.mem)
  end
  pdat.LLDLS.Fact.D .= abs.(pdat.LLDLS.Fact.D)
end
