# Formulation K2: (if regul==:classic, adds additional Regularization parmeters -ρ (top left) and δ (bottom right))
# [-Q_X - D      sqrt(X1X2)A' ] [̃x] = rhs
# [ A sqrt(X1x2)         0    ] [y]
# where Q_X = sqrt(X1X2) Q sqrt(X1X2) and D = s_l X2 + s_u X1
# and  Δ x = sqrt(X1 X2) Δ ̃x
export K2_5LDLParams

"""
Type to use the K2.5 formulation with a LDLᵀ factorization, using the package 
[`LDLFactorizations.jl`](https://github.com/JuliaSmoothOptimizers/LDLFactorizations.jl). 
The outer constructor 

    sp = K2_5LDLParams(; regul :: Symbol = :classic) 

creates a [`RipQP.SolverParams`](@ref) that should be used to create a [`RipQP.InputConfig`](@ref).
`regul = :dynamic` uses a dynamic regularization (the regularization is only added if the LDLᵀ factorization 
encounters a pivot that has a small magnitude).
`regul = :none` uses no regularization (not recommended).
"""
struct K2_5LDLParams <: SolverParams
  regul::Symbol
end

function K2_5LDLParams(; regul::Symbol = :classic)
  regul == :classic ||
    regul == :dynamic ||
    regul == :none ||
    error("regul should be :classic or :dynamic or :none")
  return K2_5LDLParams(regul)
end

mutable struct PreallocatedDataK2_5LDL{T <: Real, S} <: PreallocatedDataAugmentedLDL{T, S}
  D::S # temporary top-left diagonal
  regu::Regularization{T}
  diag_Q::SparseVector{T, Int} # Q diagonal
  K::SparseMatrixCSC{T, Int} # augmented matrix 
  K_fact::LDLFactorizations.LDLFactorization{T, Int, Int, Int} # factorized matrix
  fact_fail::Bool # true if factorization failed 
  diagind_K::Vector{Int} # diagonal indices of J
  K_scaled::Bool # true if K is scaled with X1X2
end

function PreallocatedData(
  sp::K2_5LDLParams,
  fd::QM_FloatData{T},
  id::QM_IntData,
  iconf::InputConfig{Tconf},
) where {T <: Real, Tconf <: Real}

  # init Regularization values
  D = similar(fd.c, id.nvar)
  if iconf.mode == :mono
    regu = Regularization(
      T(sqrt(eps()) * 1e5),
      T(sqrt(eps()) * 1e5),
      1e-5 * sqrt(eps(T)),
      1e0 * sqrt(eps(T)),
      sp.regul,
    )
    D .= -T(1.0e0) / 2
  else
    regu = Regularization(
      T(sqrt(eps()) * 1e5),
      T(sqrt(eps()) * 1e5),
      T(sqrt(eps(T)) * 1e0),
      T(sqrt(eps(T)) * 1e0),
      sp.regul,
    )
    D .= -T(1.0e-2)
  end
  diag_Q = get_diag_Q(fd.Q.colptr, fd.Q.rowval, fd.Q.nzval, id.nvar)
  K = create_K2(id, D, fd.Q, fd.A, diag_Q, regu)

  diagind_K = get_diag_sparseCSC(K.colptr, id.ncon + id.nvar)
  K_fact = ldl_analyze(Symmetric(K, :U))
  if regu.regul == :dynamic
    Amax = @views norm(K.nzval[diagind_K], Inf)
    regu.ρ, regu.δ = -T(eps(T)^(3 / 4)), T(eps(T)^(0.45))
    K_fact.r1, K_fact.r2 = regu.ρ, regu.δ
    K_fact.tol = Amax * T(eps(T))
    K_fact.n_d = id.nvar
  elseif regu.regul == :none
    regu.ρ, regu.δ = zero(T), zero(T)
  end
  K_fact = ldl_factorize!(Symmetric(K, :U), K_fact)
  K_fact.__factorized = true

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

function convertpad(
  ::Type{<:PreallocatedData{T}},
  pad::PreallocatedDataK2_5LDL{T_old},
  T0::DataType,
) where {T <: Real, T_old <: Real}
  pad = PreallocatedDataK2_5LDL(
    convert(Array{T}, pad.D),
    convert(Regularization{T}, pad.regu),
    convert(SparseVector{T, Int}, pad.diag_Q),
    convert(SparseMatrixCSC{T, Int}, pad.K),
    convertldl(T, pad.K_fact),
    pad.fact_fail,
    pad.diagind_K,
    pad.K_scaled,
  )

  if pad.regu.regul == :classic
    if T == Float64 && T0 == Float64
      pad.regu.ρ_min, pad.regu.δ_min = T(sqrt(eps()) * 1e-5), T(sqrt(eps()) * 1e0)
    else
      pad.regu.ρ_min, pad.regu.δ_min = T(sqrt(eps(T)) * 1e1), T(sqrt(eps(T)) * 1e1)
    end
    pad.regu.ρ /= 10
    pad.regu.δ /= 10
  elseif pad.regu.regul == :dynamic
    pad.regu.ρ, pad.regu.δ = -T(eps(T)^(3 / 4)), T(eps(T)^(0.45))
    pad.K_fact.r1, pad.K_fact.r2 = pad.regu.ρ, pad.regu.δ
  end

  return pad
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
  T0::DataType,
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
      out = update_regu_diagK2_5!(pad.regu, pad.D, itd.pdd, itd.l_pdd, itd.mean_pdd, cnts, T, T0)
    end

    # restore J for next iteration
    if pad.K_scaled
      pad.D .= one(T)
      pad.D[id.ilow] ./= sqrt.(itd.x_m_lvar)
      pad.D[id.iupp] ./= sqrt.(itd.uvar_m_x)
      lrmultilply_J!(pad.K.colptr, pad.K.rowval, pad.K.nzval, pad.D, id.nvar)
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
  T0::DataType,
) where {T <: Real}
  pad.K_scaled = false
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
    T,
    T0,
  )
  out == 1 && return out
  pad.K_scaled = true

  return out
end

function lrmultilply_J!(J_colptr, J_rowval, J_nzval, v, nvar)
  T = eltype(v)
  @inbounds @simd for i = 1:nvar
    for idx_row = J_colptr[i]:(J_colptr[i + 1] - 1)
      J_nzval[idx_row] *= v[i] * v[J_rowval[idx_row]]
    end
  end

  n = length(J_colptr)
  @inbounds @simd for i = (nvar + 1):(n - 1)
    for idx_row = J_colptr[i]:(J_colptr[i + 1] - 1)
      if J_rowval[idx_row] <= nvar
        J_nzval[idx_row] *= v[J_rowval[idx_row]] # multiply row i by v[i]
      end
    end
  end
end

# function lrmultiply_J2!(J, v)
#     lmul!(v, J)
#     rmul!(J, v)
# end

# iteration functions for the K2.5 system
function factorize_K2_5!(
  K,
  K_fact,
  D,
  diag_Q,
  diagind_K,
  regu,
  s_l,
  s_u,
  x_m_lvar,
  uvar_m_x,
  ilow,
  iupp,
  ncon,
  nvar,
  cnts,
  qp,
  T,
  T0,
)
  if regu.regul == :classic
    D .= -regu.ρ
    K.nzval[view(diagind_K, (nvar + 1):(ncon + nvar))] .= regu.δ
  else
    D .= zero(T)
  end
  D[ilow] .-= s_l ./ x_m_lvar
  D[iupp] .-= s_u ./ uvar_m_x
  D[diag_Q.nzind] .-= diag_Q.nzval
  K.nzval[view(diagind_K, 1:nvar)] = D

  D .= one(T)
  D[ilow] .*= sqrt.(x_m_lvar)
  D[iupp] .*= sqrt.(uvar_m_x)
  lrmultilply_J!(K.colptr, K.rowval, K.nzval, D, nvar)

  if regu.regul == :dynamic
    # Amax = @views norm(K.nzval[diagind_K], Inf)
    Amax = minimum(D)
    if Amax < sqrt(eps(T)) && cnts.c_pdd < 8
      if T == Float32
        # restore J for next iteration
        D .= one(T)
        D[ilow] ./= sqrt.(x_m_lvar)
        D[iupp] ./= sqrt.(uvar_m_x)
        lrmultilply_J!(K.colptr, K.rowval, K.nzval, D, nvar)
        return one(Int) # update to Float64
      elseif qp || cnts.c_pdd < 4
        cnts.c_pdd += 1
        regu.δ /= 10
        K_fact.r2 = max(sqrt(Amax), regu.δ)
        # regu.ρ /= 10
      end
    end
    K_fact.tol = min(Amax, T(eps(T)))
    K_fact = ldl_factorize!(Symmetric(K, :U), K_fact)

  elseif regu.regul == :classic
    ldl_factorize!(Symmetric(K, :U), K_fact)
    while !factorized(K_fact)
      out = update_regu_trycatch!(regu, cnts, T, T0)
      if out == 1
        # restore J for next iteration
        D .= one(T)
        D[ilow] ./= sqrt.(x_m_lvar)
        D[iupp] ./= sqrt.(uvar_m_x)
        lrmultilply_J!(K.colptr, K.rowval, K.nzval, D, nvar)
        return out
      end
      cnts.c_catch += 1
      cnts.c_catch >= 4 && return 1
      D .= -regu.ρ
      D[ilow] .-= s_l ./ x_m_lvar
      D[iupp] .-= s_u ./ uvar_m_x
      D[diag_Q.nzind] .-= diag_Q.nzval
      K.nzval[view(diagind_K, 1:nvar)] = D

      D .= one(T)
      D[ilow] .*= sqrt.(x_m_lvar)
      D[iupp] .*= sqrt.(uvar_m_x)
      K.nzval[view(diagind_K, 1:nvar)] .*= D .^ 2
      K.nzval[view(diagind_K, (nvar + 1):(ncon + nvar))] .= regu.δ
      ldl_factorize!(Symmetric(K, :U), K_fact)
    end

  else # no Regularization
    K_fact = ldl_factorize!(Symmetric(K, :U), K_fact)
  end

  return 0 # factorization succeeded
end
