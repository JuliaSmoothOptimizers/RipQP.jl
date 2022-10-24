# (A D⁻¹ Aᵀ + δI) Δy = A D⁻¹ ξ₁ + ξ₂ 
# where D = s_l (x - lvar)⁻¹ + s_u (uvar - x)⁻¹ + ρI,
# and the right hand side of K2 is rhs = [ξ₁]
#                                        [ξ₂] 
export K1CholParams

"""
Type to use the K1 formulation with a Cholesky factorization.
The outer constructor 

    sp = K1CholParams(; fact_alg = LDLFact(regul = :classic),
                      ρ0 = sqrt(eps()) * 1e5, δ0 = sqrt(eps()) * 1e5,
                      ρ_min = sqrt(eps()), δ_min = sqrt(eps()))

creates a [`RipQP.SolverParams`](@ref).
"""
mutable struct K1CholParams{T, Fact} <: NormalParams{T}
  uplo::Symbol
  fact_alg::Fact
  ρ0::T
  δ0::T
  ρ_min::T
  δ_min::T
  function K1CholParams(fact_alg::AbstractFactorization, ρ0::T, δ0::T, ρ_min::T, δ_min::T) where {T}
    return new{T, typeof(fact_alg)}(get_uplo(fact_alg), fact_alg, ρ0, δ0, ρ_min, δ_min)
  end
end

K1CholParams{T}(;
  fact_alg::AbstractFactorization = LDLFact(:classic),
  ρ0::T = (T == Float16) ? one(T) : T(sqrt(eps()) * 1e5),
  δ0::T = (T == Float16) ? one(T) : T(sqrt(eps()) * 1e5),
  ρ_min::T = sqrt(eps(T))*10,
  δ_min::T = sqrt(eps(T))*10,
) where {T} = K1CholParams(fact_alg, ρ0, δ0, ρ_min, δ_min)

K1CholParams(; kwargs...) = K1CholParams{Float64}(; kwargs...)

mutable struct PreallocatedDataK1Chol{T <: Real, S, M <: AbstractMatrix{T}, F} <:
               PreallocatedDataNormalChol{T, S}
  D::S
  regu::Regularization{T}
  K::Symmetric{T, M} # augmented matrix 
  K_fact::F # factorized matrix
  diagind_K::Vector{Int} # diagonal indices of J
  rhs::S
  ξ1tmp::S
  Aj::S
end

# outer constructor
function PreallocatedData(
  sp::K1CholParams,
  fd::QM_FloatData{T},
  id::QM_IntData,
  itd::IterData{T},
  pt::Point{T},
  iconf::InputConfig{Tconf},
) where {T <: Real, Tconf <: Real}

  # init Regularization values
  D = similar(fd.c, id.nvar)
  D .= T(1.0e0) / 2
  regu = Regularization(T(sp.ρ0), T(sp.δ0), T(sp.ρ_min), T(sp.δ_min), sp.fact_alg.regul)

  if sp.uplo == :L
    K = fd.A * fd.A' + regu.δ * I
    K = Symmetric(tril(K), sp.uplo)
  elseif sp.uplo == :U
    K = fd.A' * fd.A + regu.δ * I
    K = Symmetric(triu(K), sp.uplo)
  end
  diagind_K = get_diagind_K(K, sp.uplo)
  K_fact = init_fact(K, sp.fact_alg)
  generic_factorize!(K, K_fact)
  rhs = similar(D, id.ncon)
  ξ1tmp = similar(D)
  Aj = similar(D)

  return PreallocatedDataK1Chol(
    D,
    regu,
    K,
    K_fact,
    diagind_K,
    rhs,
    ξ1tmp,
    Aj,
  )
end

# function used to solve problems
# solver LDLFactorization
function solver!(
  dd::AbstractVector{T},
  pad::PreallocatedDataK1Chol{T},
  dda::DescentDirectionAllocs{T},
  pt::Point{T},
  itd::IterData{T},
  fd::Abstract_QM_FloatData{T},
  id::QM_IntData,
  res::AbstractResiduals{T},
  cnts::Counters,
  step::Symbol,
) where {T <: Real}
  @. pad.ξ1tmp = @views dd[1:(id.nvar)] / pad.D
  if fd.uplo == :L
    mul!(pad.rhs, fd.A, pad.ξ1tmp)
  else
    mul!(pad.rhs, fd.A', pad.ξ1tmp)
  end
  pad.rhs .+= @views dd[(id.nvar + 1):end]

  ldiv!(pad.K_fact, pad.rhs)

  if fd.uplo == :U
    @views mul!(dd[1:(id.nvar)], fd.A, pad.rhs, one(T), -one(T))
  else
    @views mul!(dd[1:(id.nvar)], fd.A', pad.rhs, one(T), -one(T))
  end
  dd[1:(id.nvar)] ./= pad.D
  dd[(id.nvar + 1):end] .= pad.rhs
  return 0
end

function update_pad!(
  pad::PreallocatedDataK1Chol{T},
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

  pad.D .= pad.regu.ρ
  @. pad.D[id.ilow] += pt.s_l / itd.x_m_lvar
  @. pad.D[id.iupp] += pt.s_u / itd.uvar_m_x

  updateK1!(pad.K, pad.D, fd.A, pad.Aj, pad.regu.δ, id.ncon)

  generic_factorize!(pad.K, pad.K_fact)

  return 0
end

# (AAᵀ)_(i,j) = ∑ A_(i,k) D_(k,k) A_(j,k)
# transpose A if uplo = U
function updateK1!(K::Symmetric{T, <:SparseMatrixCSC{T}}, D::AbstractVector{T}, A::SparseMatrixCSC{T}, Aj, δ, ncon) where {T}
  K_colptr, K_rowval, K_nzval = K.data.colptr, K.data.rowval, K.data.nzval
  A_colptr, A_rowval, A_nzval = A.colptr, A.rowval, A.nzval

  for j=1:ncon
    Aj .= zero(T)
    for kA=A_colptr[j] : (A_colptr[j+1]-1)
      k = A_rowval[kA]
      Aj[k] = A_nzval[kA] / D[k]
    end
    # Aj is col j of A divided by D

    for k2=K_colptr[j]:(K_colptr[j+1]-1)
      i = K_rowval[k2]
      i ≤ j || continue
      K_nzval[k2] = (i == j) ? δ : zero(T)
      for l=A_colptr[i] : (A_colptr[i+1]-1)
        k = A_rowval[l]
        K_nzval[k2] += A_nzval[l] * Aj[k]
      end
    end
  end
end