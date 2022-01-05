# Formulation K2: (if regul==:classic, adds additional regularization parmeters -ρ (top left) and δ (bottom right))
# [-Q - D     A' ] [x] = rhs
# [ A         0  ] [y]
export K2LDLParams

"""
Type to use the K2 formulation with a LDLᵀ factorization, using the package 
[`LDLFactorizations.jl`](https://github.com/JuliaSmoothOptimizers/LDLFactorizations.jl). 
The outer constructor 

    sp = K2LDLDenseParams(; regul = :classic, ρ0 = sqrt(eps()) * 1e5, δ0 = sqrt(eps()) * 1e5) 

creates a [`RipQP.SolverParams`](@ref) that should be used to create a [`RipQP.InputConfig`](@ref).
`regul = :dynamic` uses a dynamic regularization (the regularization is only added if the LDLᵀ factorization 
encounters a pivot that has a small magnitude).
`regul = :none` uses no regularization (not recommended).
When `regul = :classic`, the parameters `ρ0` and `δ0` are used to choose the initial regularization values.
"""
mutable struct K2LDLDenseParams <: SolverParams
  uplo::Symbol
  regul::Symbol
  ρ0::Float64
  δ0::Float64
end

function K2LDLDenseParams(;
  regul::Symbol = :classic,
  ρ0::Float64 = sqrt(eps()) * 1e5,
  δ0::Float64 = sqrt(eps()) * 1e5,
)
  regul == :classic ||
    regul == :dynamic ||
    regul == :none ||
    error("regul should be :classic or :dynamic or :none")
  uplo = :L # mandatory for LDLFactorizations
  return K2LDLDenseParams(uplo, regul, ρ0, δ0)
end

mutable struct PreallocatedDataK2LDLDense{T <: Real, S, M <: AbstractMatrix{T}} <: PreallocatedDataAugmentedLDL{T, S}
  D::S # temporary top-left diagonal
  regu::Regularization{T}
  K::M # augmented matrix 
  fact_fail::Bool # true if factorization failed 
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
    regu = Regularization(T(sp.ρ0), T(sp.δ0), 1e-5 * sqrt(eps(T)), 1e0 * sqrt(eps(T)), sp.regul)
    D .= -T(1.0e0) / 2
  else
    regu =
      Regularization(T(sp.ρ0), T(sp.δ0), T(sqrt(eps(T)) * 1e0), T(sqrt(eps(T)) * 1e0), sp.regul)
    D .= -T(1.0e-2)
  end

  if regu.regul == :none
    regu.ρ, regu.δ = zero(T), zero(T)
  end
  K = Symmetric(zeros(T, id.nvar + id.ncon, id.nvar + id.ncon), :L)
  K.data[1:id.nvar, 1:id.nvar] .= .-fd.Q .+ Diagonal(D)
  K.data[id.nvar+1: id.nvar+id.ncon, 1:id.nvar] .= fd.A
  K.data[view(diagind(K), id.nvar+1: id.nvar+id.ncon)] .= regu.δ

  ldl_dense!(K)

  return PreallocatedDataK2LDLDense(
    D,
    regu,
    K, #K
    false,
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
  T0::DataType,
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
  T0::DataType,
) where {T <: Real}
  if cnts.k != 0
    update_regu!(pad.regu)
  end

  pad.D .= -pad.regu.ρ
  pad.D[id.ilow] .-= pt.s_l ./ itd.x_m_lvar
  pad.D[id.iupp] .-= pt.s_u ./ itd.uvar_m_x
  pad.K.data[1:id.nvar, 1:id.nvar] .= .-fd.Q .+ Diagonal(pad.D)
  pad.K.data[id.nvar+1: id.nvar+id.ncon, 1:id.nvar] .= fd.A
  pad.K.data[view(diagind(pad.K), id.nvar+1: id.nvar+id.ncon)] .= pad.regu.δ

  ldl_dense!(pad.K)
  return 0
end