# (A D⁻¹ Aᵀ + δI) Δy = A D⁻¹ ξ₁ + ξ₂ 
# where D = s_l (x - lvar)⁻¹ + s_u (uvar - x)⁻¹ + ρI,
# and the right hand side of K2 is rhs = [ξ₁]
#                                        [ξ₂] 
export K1CholDenseParams

"""
Type to use the K1 formulation with a dense Cholesky factorization.
The input QuadraticModel should have `lcon .== ucon`.
The outer constructor 

    sp = K1CholDenseParams(; ρ0 = sqrt(eps()) * 1e5, δ0 = sqrt(eps()) * 1e5) 

creates a [`RipQP.SolverParams`](@ref).
"""
mutable struct K1CholDenseParams{T} <: NormalParams{T}
  uplo::Symbol
  ρ0::T
  δ0::T
end

function K1CholDenseParams{T}(; ρ0::T = T(sqrt(eps()) * 1e5), δ0::T = T(sqrt(eps()) * 1e5)) where {T}
  uplo = :L # mandatory for LDL fact
  return K1CholDenseParams(uplo, ρ0, δ0)
end

K1CholDenseParams(; kwargs...) = K1CholDenseParams{Float64}(; kwargs...)

mutable struct PreallocatedDataK1CholDense{T <: Real, S, M <: AbstractMatrix{T}} <:
               PreallocatedDataNormalChol{T, S}
  D::S # temporary top-left diagonal
  invD::Diagonal{T, S}
  AinvD::M
  rhs::S
  regu::Regularization{T}
  K::M # augmented matrix
  diagindK::StepRange{Int, Int}
  tmpldiv::S
end

# outer constructor
function PreallocatedData(
  sp::K1CholDenseParams,
  fd::QM_FloatData{T},
  id::QM_IntData,
  itd::IterData{T},
  pt::Point{T},
  iconf::InputConfig{Tconf},
) where {T <: Real, Tconf <: Real}

  # init Regularization values
  D = similar(fd.c, id.nvar)
  if iconf.mode == :mono
    regu = Regularization(T(sp.ρ0), T(sp.δ0), 1e-5 * sqrt(eps(T)), 1e0 * sqrt(eps(T)), :classic)
    D .= T(1.0e0) / 2
  else
    regu =
      Regularization(T(sp.ρ0), T(sp.δ0), T(sqrt(eps(T)) * 1e0), T(sqrt(eps(T)) * 1e0), :classic)
    D .= T(1.0e-2)
  end

  invD = Diagonal(one(T) ./ D)
  AinvD = fd.A * invD
  K = AinvD * fd.A'
  diagindK = diagind(K)
  K[diagindK] .+= regu.δ
  cholesky!(Symmetric(K))

  return PreallocatedDataK1CholDense(
    D,
    invD,
    AinvD,
    similar(D, id.ncon),
    regu,
    K, #K
    diagindK,
    similar(D, id.ncon),
  )
end

# function used to solve problems
# solver LDLFactorization
function solver!(
  dd::AbstractVector{T},
  pad::PreallocatedDataK1CholDense{T},
  dda::DescentDirectionAllocs{T},
  pt::Point{T},
  itd::IterData{T},
  fd::Abstract_QM_FloatData{T},
  id::QM_IntData,
  res::AbstractResiduals{T},
  cnts::Counters,
  T0::DataType,
  step::Symbol,
) where {T <: Real}
  pad.rhs .= @views dd[(id.nvar + 1):end]
  @views mul!(pad.rhs, pad.AinvD, dd[1:(id.nvar)], one(T), one(T))

  ldiv!(pad.tmpldiv, UpperTriangular(pad.K)', pad.rhs)
  ldiv!(pad.rhs, UpperTriangular(pad.K), pad.tmpldiv)

  @views mul!(dd[1:(id.nvar)], fd.A', pad.rhs, one(T), -one(T))
  dd[1:(id.nvar)] ./= pad.D
  dd[(id.nvar + 1):end] .= pad.rhs
  return 0
end

function update_pad!(
  pad::PreallocatedDataK1CholDense{T},
  dda::DescentDirectionAllocs{T},
  pt::Point{T},
  itd::IterData{T},
  fd::Abstract_QM_FloatData{T},
  id::QM_IntData,
  res::AbstractResiduals{T},
  cnts::Counters,
  T0::DataType,
) where {T <: Real}
  if cnts.k != 0
    update_regu!(pad.regu)
  end

  pad.D .= pad.regu.ρ
  pad.D[id.ilow] .+= pt.s_l ./ itd.x_m_lvar
  pad.D[id.iupp] .+= pt.s_u ./ itd.uvar_m_x
  pad.invD.diag .= one(T) ./ pad.D
  mul!(pad.AinvD, fd.A, pad.invD)
  mul!(pad.K, pad.AinvD, fd.A')
  pad.K[pad.diagindK] .+= pad.regu.δ
  cholesky!(Symmetric(pad.K))

  return 0
end

function convertpad(
  ::Type{<:PreallocatedData{T}},
  pad::PreallocatedDataK1CholDense{T0, S0, M0},
  sp_old::K1CholDenseParams,
  sp_new::Union{Nothing, K1CholDenseParams},
  id::QM_IntData,
  fd::Abstract_QM_FloatData,
  T02::DataType,
) where {T <: Real, T0 <: Real, S0, M0}
  S = change_vector_eltype(S0, T)
  pad = PreallocatedDataK1CholDense(
    convert(S, pad.D),
    Diagonal(convert(S, pad.invD.diag)),
    convert_mat(pad.AinvD, T),
    convert(S, pad.rhs),
    convert(Regularization{T}, pad.regu),
    convert_mat(pad.K, T),
    pad.diagindK,
    convert(S, pad.tmpldiv),
  )

  if T == Float64 && T0 == Float64
    pad.regu.ρ_min, pad.regu.δ_min = T(sqrt(eps()) * 1e0), T(sqrt(eps()) * 1e0)
  else
    pad.regu.ρ_min, pad.regu.δ_min = T(sqrt(eps(T)) * 1e1), T(sqrt(eps(T)) * 1e1)
  end

  return pad
end
