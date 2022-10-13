# Formulation K2: (if regul==:classic, adds additional Regularization parmeters -ρ (top left) and δ (bottom right))
# [-Q_X - D      sqrt(X1X2)A' ] [̃x] = rhs
# [ A sqrt(X1x2)         0    ] [y]
# where Q_X = sqrt(X1X2) Q sqrt(X1X2) and D = s_l X2 + s_u X1
# and  Δ x = sqrt(X1 X2) Δ ̃x
export K2_5LDLParams

"""
Type to use the K2.5 formulation with a LDLᵀ factorization.
The package [`LDLFactorizations.jl`](https://github.com/JuliaSmoothOptimizers/LDLFactorizations.jl)
is used by default.
The outer constructor 

    sp = K2_5LDLParams(; fact_alg = LDLFact(regul = :classic), ρ0 = sqrt(eps()) * 1e5, δ0 = sqrt(eps()) * 1e5)

creates a [`RipQP.SolverParams`](@ref).
`regul = :dynamic` uses a dynamic regularization (the regularization is only added if the LDLᵀ factorization 
encounters a pivot that has a small magnitude).
`regul = :none` uses no regularization (not recommended).
When `regul = :classic`, the parameters `ρ0` and `δ0` are used to choose the initial regularization values.
`fact_alg` should be a [`RipQP.AbstractFactorization`](@ref).
"""
mutable struct K2_5LDLParams{T, Fact} <: AugmentedParams{T}
  uplo::Symbol
  fact_alg::Fact
  ρ0::T
  δ0::T
  ρ_min::T
  δ_min::T
  function K2_5LDLParams(
    fact_alg::AbstractFactorization,
    ρ0::T,
    δ0::T,
    ρ_min::T,
    δ_min::T,
  ) where {T}
    return new{T, typeof(fact_alg)}(get_uplo(fact_alg), fact_alg, ρ0, δ0, ρ_min, δ_min)
  end
end

K2_5LDLParams{T}(;
  fact_alg::AbstractFactorization = LDLFact(:classic),
  ρ0::T = (T == Float64) ? sqrt(eps()) * 1e5 : one(T),
  δ0::T = (T == Float64) ? sqrt(eps()) * 1e5 : one(T),
  ρ_min::T = (T == Float64) ? 1e-5 * sqrt(eps()) : sqrt(eps(T)),
  δ_min::T = (T == Float64) ? 1e0 * sqrt(eps()) : sqrt(eps(T)),
) where {T} = K2_5LDLParams(fact_alg, ρ0, δ0, ρ_min, δ_min)

K2_5LDLParams(; kwargs...) = K2_5LDLParams{Float64}(; kwargs...)

mutable struct PreallocatedDataK2_5LDL{T <: Real, S, M, F <: FactorizationData{T}} <:
               PreallocatedDataAugmentedLDL{T, S}
  D::S # temporary top-left diagonal
  regu::Regularization{T}
  diag_Q::SparseVector{T, Int} # Q diagonal
  K::Symmetric{T, M} # augmented matrix 
  K_fact::F # factorized matrix
  fact_fail::Bool # true if factorization failed 
  diagind_K::Vector{Int} # diagonal indices of J
  K_scaled::Bool # true if K is scaled with X1X2
end

solver_name(pad::PreallocatedDataK2_5LDL) =
  string(string(typeof(pad).name.name)[17:end], " with $(typeof(pad.K_fact).name.name)")

function PreallocatedData(
  sp::K2_5LDLParams,
  fd::QM_FloatData{T},
  id::QM_IntData,
  itd::IterData{T},
  pt::Point{T},
  iconf::InputConfig{Tconf},
) where {T <: Real, Tconf <: Real}

  # init Regularization values
  D = similar(fd.c, id.nvar)
  D .= -T(1.0e0) / 2
  regu = Regularization(T(sp.ρ0), T(sp.δ0), T(sp.ρ_min), T(sp.δ_min), sp.fact_alg.regul)
  diag_Q = get_diag_Q(fd.Q)
  K = create_K2(id, D, fd.Q.data, fd.A, diag_Q, regu, sp.uplo, T)

  diagind_K = get_diagind_K(K, sp.uplo)
  K_fact = init_fact(K, sp.fact_alg)
  if regu.regul == :dynamic
    Amax = @views norm(K.data.nzval[diagind_K], Inf)
    regu.ρ, regu.δ = T(eps(T)^(3 / 4)), T(eps(T)^(0.45))
    K_fact.LDL.r1, K_fact.LDL.r2 = -regu.ρ, regu.δ
    K_fact.LDL.tol = Amax * T(eps(T))
    K_fact.LDL.n_d = id.nvar
  elseif regu.regul == :none
    regu.ρ, regu.δ = zero(T), zero(T)
  end
  generic_factorize!(K, K_fact)

  return PreallocatedDataK2_5LDL(
    D,
    regu,
    diag_Q, #diag_Q
    K, #K
    K_fact, #K_fact
    false,
    diagind_K, #diagind_K
    false,
  )
end

# solver LDLFactorization
function solver!(
  dd::AbstractVector{T},
  pad::PreallocatedDataK2_5LDL{T},
  dda::DescentDirectionAllocs{T},
  pt::Point{T},
  itd::IterData{T},
  fd::Abstract_QM_FloatData{T},
  id::QM_IntData,
  res::AbstractResiduals{T},
  cnts::Counters,
  step::Symbol,
) where {T <: Real}
  if pad.K_scaled
    dd[1:(id.nvar)] .*= pad.D
    ldiv!(pad.K_fact, dd)
    dd[1:(id.nvar)] .*= pad.D
  else
    ldiv!(pad.K_fact, dd)
  end

  if step == :cc || step == :IPF  # update regularization and restore K. Cannot be done in update_pad since x-lvar and uvar-x will change.
    out = 0
    if pad.regu.regul == :classic # update ρ and δ values, check K diag magnitude 
      out = update_regu_diagK2_5!(pad.regu, pad.D, itd.pdd, itd.l_pdd, itd.mean_pdd, cnts)
    end

    # restore J for next iteration
    if pad.K_scaled
      pad.D .= one(T)
      @. pad.D[id.ilow] /= sqrt(itd.x_m_lvar)
      @. pad.D[id.iupp] /= sqrt(itd.uvar_m_x)
      lrmultilply_K!(pad.K, pad.D, id.nvar)
      pad.K_scaled = false
    end
    out == 1 && return out
  end
  return 0
end

function update_pad!(
  pad::PreallocatedDataK2_5LDL{T},
  dda::DescentDirectionAllocs{T},
  pt::Point{T},
  itd::IterData{T},
  fd::Abstract_QM_FloatData{T},
  id::QM_IntData,
  res::AbstractResiduals{T},
  cnts::Counters,
) where {T <: Real}
  pad.K_scaled = false
  update_K!(
    pad.K,
    pad.D,
    pad.regu,
    pt.s_l,
    pt.s_u,
    itd.x_m_lvar,
    itd.uvar_m_x,
    id.ilow,
    id.iupp,
    pad.diag_Q,
    pad.diagind_K,
    id.nvar,
    id.ncon,
  )
  out = factorize_K2_5!(
    pad.K,
    pad.K_fact,
    pad.D,
    pad.diag_Q,
    pad.diagind_K,
    pad.regu,
    pt.s_l,
    pt.s_u,
    itd.x_m_lvar,
    itd.uvar_m_x,
    id.ilow,
    id.iupp,
    id.ncon,
    id.nvar,
    cnts,
    itd.qp,
  )
  out == 1 && return out
  pad.K_scaled = true

  return out
end

function lrmultilply_K_CSC!(K_colptr, K_rowval, K_nzval::Vector{T}, v::Vector{T}, nvar) where {T}
  @inbounds @simd for i = 1:nvar
    for idx_row = K_colptr[i]:(K_colptr[i + 1] - 1)
      K_nzval[idx_row] *= v[i] * v[K_rowval[idx_row]]
    end
  end

  n = length(K_colptr)
  @inbounds @simd for i = (nvar + 1):(n - 1)
    for idx_row = K_colptr[i]:(K_colptr[i + 1] - 1)
      if K_rowval[idx_row] <= nvar
        K_nzval[idx_row] *= v[K_rowval[idx_row]] # multiply row i by v[i]
      end
    end
  end
end

lrmultilply_K!(K::Symmetric{T, SparseMatrixCSC{T, Int}}, v::Vector{T}, nvar) where {T} =
  lrmultilply_K_CSC!(K.data.colptr, K.data.rowval, K.data.nzval, v, nvar)

function lrmultilply_K!(K::Symmetric{T, SparseMatrixCOO{T, Int}}, v::Vector{T}, nvar) where {T}
  lmul!(v, K.data)
  rmul!(K.data, v)
end

function X1X2_to_D!(
  D::AbstractVector{T},
  x_m_lvar::AbstractVector{T},
  uvar_m_x::AbstractVector{T},
  ilow,
  iupp,
) where {T}
  D .= one(T)
  D[ilow] .*= sqrt.(x_m_lvar)
  D[iupp] .*= sqrt.(uvar_m_x)
end

function update_Dsquare_diag_K11!(K::SparseMatrixCSC, D, diagind_K, nvar)
  K.data.nzval[view(diagind_K, 1:nvar)] .*= D .^ 2
end

function update_Dsquare_diag_K11!(K::SparseMatrixCOO, D, diagind_K, nvar)
  K.data.vals[view(diagind_K, 1:nvar)] .*= D .^ 2
end

# iteration functions for the K2.5 system
function factorize_K2_5!(
  K::Symmetric{T},
  K_fact::FactorizationData{T},
  D::AbstractVector{T},
  diag_Q::AbstractSparseVector{T},
  diagind_K,
  regu::Regularization{T},
  s_l::AbstractVector{T},
  s_u::AbstractVector{T},
  x_m_lvar::AbstractVector{T},
  uvar_m_x::AbstractVector{T},
  ilow,
  iupp,
  ncon,
  nvar,
  cnts::Counters,
  qp::Bool,
) where {T}
  X1X2_to_D!(D, x_m_lvar, uvar_m_x, ilow, iupp)
  lrmultilply_K!(K, D, nvar)
  if regu.regul == :dynamic
    # Amax = @views norm(K.nzval[diagind_K], Inf)
    Amax = minimum(D)
    if Amax < sqrt(eps(T)) && cnts.c_pdd < 8
      if cnts.last_sp
        # restore K for next iteration
        X1X2_to_D!(D, x_m_lvar, uvar_m_x, ilow, iupp)
        lrmultilply_K!(K, D, nvar)
        return one(Int) # update to Float64
      elseif qp || cnts.c_pdd < 4
        cnts.c_pdd += 1
        regu.δ /= 10
        K_fact.LDL.r2 = max(sqrt(Amax), regu.δ)
        # regu.ρ /= 10
      end
    end
    K_fact.LDL.tol = min(Amax, T(eps(T)))
    generic_factorize!(K, K_fact)

  elseif regu.regul == :classic
    generic_factorize!(K, K_fact)
    while !RipQP.factorized(K_fact)
      out = update_regu_trycatch!(regu, cnts)
      if out == 1
        # restore J for next iteration
        X1X2_to_D!(D, x_m_lvar, uvar_m_x, ilow, iupp)
        lrmultilply_K!(K, D, nvar)
        return out
      end
      cnts.c_catch += 1
      cnts.c_catch >= 4 && return 1
      update_K!(K, D, regu, s_l, s_u, x_m_lvar, uvar_m_x, ilow, iupp, diag_Q, diagind_K, nvar, ncon)

      X1X2_to_D!(D, x_m_lvar, uvar_m_x, ilow, iupp)
      K.data.nzval[view(diagind_K, 1:nvar)] .*= D .^ 2
      update_diag_K22!(K, regu.δ, diagind_K, nvar, ncon)
      generic_factorize!(K, K_fact)
    end

  else # no Regularization
    generic_factorize!(K, K_fact)
  end

  return 0 # factorization succeeded
end
