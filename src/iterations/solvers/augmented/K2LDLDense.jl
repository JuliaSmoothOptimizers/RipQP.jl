# Formulation K2: (if regul==:classic, adds additional regularization parmeters -ρ (top left) and δ (bottom right))
# [-Q - D     A' ] [x] = rhs
# [ A         0  ] [y]
export K2LDLDenseParams

"""
Type to use the K2 formulation with a LDLᵀ factorization. 
The outer constructor 

    sp = K2LDLDenseParams(; ρ0 = sqrt(eps()) * 1e5, δ0 = sqrt(eps()) * 1e5) 

creates a [`RipQP.SolverParams`](@ref).
"""
mutable struct K2LDLDenseParams{T} <: AugmentedParams{T}
  uplo::Symbol
  fact_alg::Symbol
  ρ0::T
  δ0::T
end

function K2LDLDenseParams(;
  fact_alg::Symbol = :bunchkaufman,
  ρ0::Float64 = sqrt(eps()) * 1e5,
  δ0::Float64 = sqrt(eps()) * 1e5,
)
  uplo = :L # mandatory for LDL fact
  return K2LDLDenseParams(uplo, fact_alg, ρ0, δ0)
end

mutable struct PreallocatedDataK2LDLDense{T <: Real, S, M <: AbstractMatrix{T}} <:
               PreallocatedDataAugmentedLDL{T, S}
  D::S # temporary top-left diagonal
  regu::Regularization{T}
  K::M # augmented matrix
  fact_alg::Symbol
end

# outer constructor
function PreallocatedData(
  sp::K2LDLDenseParams,
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
    D .= -T(1.0e0) / 2
  else
    regu =
      Regularization(T(sp.ρ0), T(sp.δ0), T(sqrt(eps(T)) * 1e0), T(sqrt(eps(T)) * 1e0), :classic)
    D .= -T(1.0e-2)
  end

  K = Symmetric(zeros(T, id.nvar + id.ncon, id.nvar + id.ncon), :L)
  K.data[1:(id.nvar), 1:(id.nvar)] .= .-fd.Q.data .+ Diagonal(D)
  K.data[(id.nvar + 1):(id.nvar + id.ncon), 1:(id.nvar)] .= fd.A
  K.data[view(diagind(K), (id.nvar + 1):(id.nvar + id.ncon))] .= regu.δ

  if sp.fact_alg == :bunchkaufman
    bunchkaufman!(K)
  elseif sp.fact_alg == :ldl
    ldl_dense!(K)
  end

  return PreallocatedDataK2LDLDense(
    D,
    regu,
    K, #K
    sp.fact_alg,
  )
end

# function used to solve problems
# solver LDLFactorization
function solver!(
  dd::AbstractVector{T},
  pad::PreallocatedDataK2LDLDense{T},
  dda::DescentDirectionAllocs{T},
  pt::Point{T},
  itd::IterData{T},
  fd::Abstract_QM_FloatData{T},
  id::QM_IntData,
  res::AbstractResiduals{T},
  cnts::Counters,
  step::Symbol,
) where {T <: Real}
  ldiv_dense!(pad.K, dd)
  return 0
end

function update_pad!(
  pad::PreallocatedDataK2LDLDense{T},
  dda::DescentDirectionAllocs{T},
  pt::Point{T},
  itd::IterData{T},
  fd::Abstract_QM_FloatData{T},
  id::QM_IntData,
  res::AbstractResiduals{T},
  cnts::Counters,
) where {T <: Real}
  if cnts.k != 0
    update_regu!(pad.regu)
  end

  pad.D .= -pad.regu.ρ
  pad.D[id.ilow] .-= pt.s_l ./ itd.x_m_lvar
  pad.D[id.iupp] .-= pt.s_u ./ itd.uvar_m_x
  pad.K.data[1:(id.nvar), 1:(id.nvar)] .= .-fd.Q.data .+ Diagonal(pad.D)
  pad.K.data[(id.nvar + 1):(id.nvar + id.ncon), 1:(id.nvar)] .= fd.A
  pad.K.data[(id.nvar + 1):(id.nvar + id.ncon), (id.nvar + 1):(id.nvar + id.ncon)] .= zero(T)
  pad.K.data[view(diagind(pad.K), (id.nvar + 1):(id.nvar + id.ncon))] .= pad.regu.δ

  if pad.fact_alg == :bunchkaufman
    bunchkaufman!(pad.K)
  elseif pad.fact_alg == :ldl
    ldl_dense!(pad.K)
  end

  return 0
end
