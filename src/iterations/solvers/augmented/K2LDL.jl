# Formulation K2: (if regul==:classic, adds additional regularization parmeters -ρ (top left) and δ (bottom right))
# [-Q - D     A' ] [x] = rhs
# [ A         0  ] [y]
export K2LDLParams

"""
Type to use the K2 formulation with a LDLᵀ factorization.
The package [`LDLFactorizations.jl`](https://github.com/JuliaSmoothOptimizers/LDLFactorizations.jl)
is used by default.
The outer constructor 

    sp = K2LDLParams(; fact_alg = LDLFact(regul = :classic),
                     ρ0 = sqrt(eps()) * 1e5, δ0 = sqrt(eps()) * 1e5) 

creates a [`RipQP.SolverParams`](@ref).
`regul = :dynamic` uses a dynamic regularization (the regularization is only added if the LDLᵀ factorization 
encounters a pivot that has a small magnitude).
`regul = :none` uses no regularization (not recommended).
When `regul = :classic`, the parameters `ρ0` and `δ0` are used to choose the initial regularization values.
`fact_alg` should be a [`RipQP.AbstractFactorization`](@ref).
"""
mutable struct K2LDLParams{T, Fact} <: AugmentedParams{T}
  uplo::Symbol
  fact_alg::Fact
  ρ0::T
  δ0::T
  ρ_min::T
  δ_min::T
  function K2LDLParams(fact_alg::AbstractFactorization, ρ0::T, δ0::T, ρ_min::T, δ_min::T) where {T}
    return new{T, typeof(fact_alg)}(get_uplo(fact_alg), fact_alg, ρ0, δ0, ρ_min, δ_min)
  end
end

K2LDLParams{T}(;
  fact_alg::AbstractFactorization = LDLFact(:classic),
  ρ0::T = (T == Float16) ? one(T) : T(sqrt(eps()) * 1e5),
  δ0::T = (T == Float16) ? one(T) : T(sqrt(eps()) * 1e5),
  ρ_min::T = (T == Float64) ? 1e-5 * sqrt(eps()) : sqrt(eps(T)),
  δ_min::T = (T == Float64) ? 1e0 * sqrt(eps()) : sqrt(eps(T)),
) where {T} = K2LDLParams(fact_alg, ρ0, δ0, ρ_min, δ_min)

K2LDLParams(; kwargs...) = K2LDLParams{Float64}(; kwargs...)

mutable struct PreallocatedDataK2LDL{T <: Real, S, F, M <: AbstractMatrix{T}} <:
               PreallocatedDataAugmentedLDL{T, S}
  D::S # temporary top-left diagonal
  regu::Regularization{T}
  diag_Q::SparseVector{T, Int} # Q diagonal
  K::Symmetric{T, M} # augmented matrix 
  K_fact::F # factorized matrix
  fact_fail::Bool # true if factorization failed 
  diagind_K::Vector{Int} # diagonal indices of J
end

solver_name(pad::PreallocatedDataK2LDL) =
  string(string(typeof(pad).name.name)[17:end], " with $(typeof(pad.K_fact).name.name)")

# outer constructor
function PreallocatedData(
  sp::K2LDLParams,
  fd::QM_FloatData{T},
  id::QM_IntData,
  itd::IterData{T},
  pt::Point{T},
  iconf::InputConfig{Tconf},
) where {T <: Real, Tconf <: Real}

  # init Regularization values
  D = similar(fd.c, id.nvar)
  D .= -T(1.0e0) / 2
  if iconf.mode == :mono
    regu = Regularization(T(sp.ρ0), T(sp.δ0), T(sp.ρ_min), T(sp.δ_min), sp.fact_alg.regul)
  else
    regu = Regularization(T(sp.ρ0), T(sp.δ0), sqrt(eps(T)), sqrt(eps(T)), sp.fact_alg.regul)
  end
  K, diagind_K, diag_Q = get_K2_matrixdata(id, D, fd.Q, fd.A, regu, sp.uplo, T)

  K_fact = init_fact(K, sp.fact_alg)
  if regu.regul == :dynamic || regu.regul == :hybrid
    Amax = @views norm(K.data.nzval[diagind_K], Inf)
    # regu.ρ, regu.δ = T(eps(T)^(3 / 4)), T(eps(T)^(0.45)) # ρ and δ kept for hybrid mode
    K_fact.LDL.r1, K_fact.LDL.r2 = -T(eps(T)^(3 / 4)), T(eps(T)^(0.45))
    K_fact.LDL.tol = Amax * T(eps(T))
    K_fact.LDL.n_d = id.nvar
  elseif regu.regul == :none
    regu.ρ, regu.δ = zero(T), zero(T)
  end
  generic_factorize!(K, K_fact)

  return PreallocatedDataK2LDL(
    D,
    regu,
    diag_Q, #diag_Q
    K, #K
    K_fact, #K_fact
    false,
    diagind_K, #diagind_K
  )
end

# function used to solve problems
# solver LDLFactorization
function solver!(
  dd::AbstractVector{T},
  pad::PreallocatedDataK2LDL{T},
  dda::DescentDirectionAllocs{T},
  pt::Point{T},
  itd::IterData{T},
  fd::Abstract_QM_FloatData{T},
  id::QM_IntData,
  res::AbstractResiduals{T},
  cnts::Counters,
  step::Symbol,
) where {T <: Real}
  ldiv!(pad.K_fact, dd)
  return 0
end

function update_pad!(
  pad::PreallocatedDataK2LDL{T},
  dda::DescentDirectionAllocs{T},
  pt::Point{T},
  itd::IterData{T},
  fd::Abstract_QM_FloatData{T},
  id::QM_IntData,
  res::AbstractResiduals{T},
  cnts::Counters,
) where {T <: Real}
  if (pad.regu.regul == :classic || pad.regu.regul == :hybrid) && cnts.k != 0
    # update ρ and δ values, check K diag magnitude 
    out = update_regu_diagK2!(pad.regu, pad.K, pad.diagind_K, id.nvar, itd, cnts)
    out == 1 && return out
  end

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
  out = factorize_K2!(
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
  ) # update D and factorize K

  if out == 1
    pad.fact_fail = true
    return out
  end

  return 0
end

# Init functions for the K2 system when uplo = U
function fill_K2_U!(
  K_colptr,
  K_rowval,
  K_nzval,
  D,
  Q_colptr,
  Q_rowval,
  Q_nzval,
  A_colptr,
  A_rowval,
  A_nzval,
  δ,
  ncon,
  nvar,
  regul,
)
  added_coeffs_diag = 0 # we add coefficients that do not appear in Q in position i,i if Q[i,i] = 0
  nnz_Q = length(Q_rowval)
  @inbounds for j = 1:nvar  # Q coeffs, tmp diag coefs. 
    K_colptr[j + 1] = Q_colptr[j + 1] + added_coeffs_diag # add previously added diagonal elements
    for k = Q_colptr[j]:(Q_colptr[j + 1] - 1)
      nz_idx = k + added_coeffs_diag
      K_rowval[nz_idx] = Q_rowval[k]
      K_nzval[nz_idx] = -Q_nzval[k]
    end
    k = Q_colptr[j + 1] - 1
    if k ≤ nnz_Q && k != 0 && Q_rowval[k] == j
      nz_idx = K_colptr[j + 1] - 1
      K_rowval[nz_idx] = j
      K_nzval[nz_idx] = D[j] - Q_nzval[k]
    else
      added_coeffs_diag += 1
      K_colptr[j + 1] += 1
      nz_idx = K_colptr[j + 1] - 1
      K_rowval[nz_idx] = j
      K_nzval[nz_idx] = D[j]
    end
  end

  countsum = K_colptr[nvar + 1] # current value of K_colptr[Q.n+j+1]
  nnz_top_left = countsum # number of coefficients + 1 already added
  @inbounds for j = 1:ncon
    countsum += A_colptr[j + 1] - A_colptr[j]
    if (regul == :classic || regul == :hybrid) && δ > 0
      countsum += 1
    end
    K_colptr[nvar + j + 1] = countsum
    for k = A_colptr[j]:(A_colptr[j + 1] - 1)
      nz_idx =
        ((regul == :classic || regul == :hybrid) && δ > 0) ? k + nnz_top_left + j - 2 :
        k + nnz_top_left - 1
      K_rowval[nz_idx] = A_rowval[k]
      K_nzval[nz_idx] = A_nzval[k]
    end
    if (regul == :classic || regul == :hybrid) && δ > 0
      nz_idx = K_colptr[nvar + j + 1] - 1
      K_rowval[nz_idx] = nvar + j
      K_nzval[nz_idx] = δ
    end
  end
end

# Init functions for the K2 system when uplo = U
function fill_K2_L!(
  K_colptr,
  K_rowval,
  K_nzval,
  D,
  Q_colptr,
  Q_rowval,
  Q_nzval,
  A_colptr,
  A_rowval,
  A_nzval,
  δ,
  ncon,
  nvar,
  regul,
)
  added_coeffs_diag = 0 # we add coefficients that do not appear in Q in position i,i if Q[i,i] = 0
  nnz_Q = length(Q_rowval)
  @inbounds for j = 1:nvar  # Q coeffs, tmp diag coefs. 
    k = Q_colptr[j]
    K_deb_ptr = K_colptr[j]
    if k ≤ nnz_Q && k != 0 && Q_rowval[k] == j
      added_diagj = false
      K_rowval[K_deb_ptr] = j
      K_nzval[K_deb_ptr] = D[j] - Q_nzval[k]
    else
      added_diagj = true
      added_coeffs_diag += 1
      K_rowval[K_deb_ptr] = j
      K_nzval[K_deb_ptr] = D[j]
    end
    nb_col_elements = 1
    deb = added_diagj ? Q_colptr[j] : (Q_colptr[j] + 1) # if already addded a new Kjj, do not add Qjj
    for k = deb:(Q_colptr[j + 1] - 1)
      nz_idx = K_deb_ptr + nb_col_elements
      nb_col_elements += 1
      K_rowval[nz_idx] = Q_rowval[k]
      K_nzval[nz_idx] = -Q_nzval[k]
    end
    # nb_elemsj_bloc11 = nb_col_elements
    for k = A_colptr[j]:(A_colptr[j + 1] - 1)
      nz_idx = K_deb_ptr + nb_col_elements
      nb_col_elements += 1
      K_rowval[nz_idx] = A_rowval[k] + nvar
      K_nzval[nz_idx] = A_nzval[k]
    end
    K_colptr[j + 1] = K_colptr[j] + nb_col_elements
  end

  if (regul == :classic || regul == :hybrid) && δ > 0
    @inbounds for j = 1:ncon
      K_prevptr = K_colptr[nvar + j]
      K_colptr[nvar + j + 1] = K_prevptr + 1
      K_rowval[K_prevptr] = nvar + j
      K_nzval[K_prevptr] = δ
    end
  end
end

function create_K2(
  id::QM_IntData,
  D::AbstractVector,
  Q::SparseMatrixCSC,
  A::SparseMatrixCSC,
  diag_Q::SparseVector,
  regu::Regularization,
  uplo::Symbol,
  ::Type{T},
) where {T}
  # for classic regul only
  n_nz = length(D) - length(diag_Q.nzind) + length(A.nzval) + length(Q.nzval)
  if (regu.regul == :classic || regu.regul == :hybrid) && regu.δ > 0
    n_nz += id.ncon
  end
  K_colptr = Vector{Int}(undef, id.ncon + id.nvar + 1)
  K_colptr[1] = 1
  K_rowval = Vector{Int}(undef, n_nz)
  K_nzval = Vector{T}(undef, n_nz)

  if uplo == :L
    # [-Q -D    0]
    # [  A     δI]
    fill_K2_L!(
      K_colptr,
      K_rowval,
      K_nzval,
      D,
      Q.colptr,
      Q.rowval,
      Q.nzval,
      A.colptr,
      A.rowval,
      A.nzval,
      regu.δ,
      id.ncon,
      id.nvar,
      regu.regul,
    )
  else
    # [-Q -D    A]
    # [0       δI]
    fill_K2_U!(
      K_colptr,
      K_rowval,
      K_nzval,
      D,
      Q.colptr,
      Q.rowval,
      Q.nzval,
      A.colptr,
      A.rowval,
      A.nzval,
      regu.δ,
      id.ncon,
      id.nvar,
      regu.regul,
    )
  end
  return Symmetric(
    SparseMatrixCSC{T, Int}(id.ncon + id.nvar, id.ncon + id.nvar, K_colptr, K_rowval, K_nzval),
    uplo,
  )
end

function create_K2(
  id::QM_IntData,
  D::AbstractVector,
  Q::SparseMatrixCOO,
  A::SparseMatrixCOO,
  diag_Q::SparseVector,
  regu::Regularization,
  uplo::Symbol,
  ::Type{T},
) where {T}
  nvar, ncon = id.nvar, id.ncon
  δ = regu.δ
  nnz_Q, nnz_A = nnz(Q), nnz(A)
  Qrows, Qcols, Qvals = Q.rows, Q.cols, Q.vals
  Arows, Acols, Avals = A.rows, A.cols, A.vals
  nnz_tot = nnz_A + nnz_Q + nvar + id.ncon - length(diag_Q.nzind)
  rows = Vector{Int}(undef, nnz_tot)
  cols = Vector{Int}(undef, nnz_tot)
  vals = Vector{T}(undef, nnz_tot)
  kQ, kA = 1, 1
  current_col = 1
  added_diag_col = false # true if K[j, j] has been filled

  for k = 1:nnz_tot
    if current_col > nvar
      rows[k] = current_col
      cols[k] = current_col
      vals[k] = δ
    elseif !added_diag_col # add diagonal
      rows[k] = current_col
      cols[k] = current_col
      if kQ ≤ nnz_Q && Qrows[kQ] == Qcols[kQ] == current_col
        vals[k] = -Qvals[kQ] + D[current_col]
        kQ += 1
      else
        vals[k] = D[current_col]
      end
      added_diag_col = true
    elseif kQ ≤ nnz_Q && Qcols[kQ] == current_col
      rows[k] = Qrows[kQ]
      cols[k] = Qcols[kQ]
      vals[k] = -Qvals[kQ]
      kQ += 1
    elseif kA ≤ nnz_A && Acols[kA] == current_col
      rows[k] = Arows[kA] + nvar
      cols[k] = Acols[kA]
      vals[k] = Avals[kA]
      kA += 1
    end
    if (kQ > nnz_Q || Qcols[kQ] != current_col) && (kA > nnz_A || Acols[kA] != current_col)
      current_col += 1
      added_diag_col = false
    end
  end
  return Symmetric(SparseMatrixCOO(nvar + ncon, nvar + ncon, rows, cols, vals), uplo)
end

function get_K2_matrixdata(
  id::QM_IntData,
  D::AbstractVector,
  Q::Symmetric,
  A::AbstractSparseMatrix,
  regu::Regularization,
  uplo::Symbol,
  ::Type{T},
) where {T}
  diag_Q = get_diag_Q(Q)
  K = create_K2(id, D, Q.data, A, diag_Q, regu, uplo, T)
  diagind_K = get_diagind_K(K, uplo)
  return K, diagind_K, diag_Q
end

function update_D!(
  D::AbstractVector{T},
  x_m_lvar::AbstractVector{T},
  uvar_m_x::AbstractVector{T},
  s_l::AbstractVector{T},
  s_u::AbstractVector{T},
  ρ::T,
  ilow,
  iupp,
) where {T}
  D .= -ρ
  @. D[ilow] -= s_l / x_m_lvar
  @. D[iupp] -= s_u / uvar_m_x
end

function update_diag_K11!(K::Symmetric{T, <:SparseMatrixCSC}, D, diagind_K, nvar) where {T}
  K.data.nzval[view(diagind_K, 1:nvar)] = D
end

function update_diag_K11!(K::Symmetric{T, <:SparseMatrixCOO}, D, diagind_K, nvar) where {T}
  K.data.vals[view(diagind_K, 1:nvar)] = D
end

function update_diag_K22!(K::Symmetric{T, <:SparseMatrixCSC}, δ, diagind_K, nvar, ncon) where {T}
  K.data.nzval[view(diagind_K, (nvar + 1):(ncon + nvar))] .= δ
end

function update_diag_K22!(K::Symmetric{T, <:SparseMatrixCOO}, δ, diagind_K, nvar, ncon) where {T}
  K.data.vals[view(diagind_K, (nvar + 1):(ncon + nvar))] .= δ
end

function update_K!(
  K::Symmetric{T},
  D::AbstractVector{T},
  regu::Regularization,
  s_l::AbstractVector{T},
  s_u::AbstractVector{T},
  x_m_lvar::AbstractVector{T},
  uvar_m_x::AbstractVector{T},
  ilow,
  iupp,
  diag_Q::AbstractSparseVector{T},
  diagind_K,
  nvar,
  ncon,
) where {T}
  if regu.regul == :classic || regu.regul == :hybrid
    ρ = regu.ρ
    if regu.δ > 0
      update_diag_K22!(K, regu.δ, diagind_K, nvar, ncon)
    end
  else
    ρ = zero(T)
  end
  update_D!(D, x_m_lvar, uvar_m_x, s_l, s_u, ρ, ilow, iupp)
  D[diag_Q.nzind] .-= diag_Q.nzval
  update_diag_K11!(K, D, diagind_K, nvar)
end

function update_K_dynamic!(
  K::Symmetric{T},
  K_fact::LDLFactorizations.LDLFactorization{Tlow},
  regu::Regularization{Tlow},
  diagind_K,
  cnts::Counters,
  qp::Bool,
) where {T, Tlow}
  Amax = @views norm(K.data.nzval[diagind_K], Inf)
  if Amax > T(1e6) / K_fact.r2 && cnts.c_pdd < 8
    if Tlow == Float32 && regu.regul == :dynamic
      return one(Int) # update to Float64
    elseif (qp || cnts.c_pdd < 4) && regu.regul == :dynamic
      cnts.c_pdd += 1
      regu.δ /= 10
      K_fact.r2 = regu.δ
    end
  end
  K_fact.tol = Tlow(Amax * eps(Tlow))
end

# iteration functions for the K2 system
function factorize_K2!(
  K::Symmetric{T},
  K_fact::FactorizationData{Tlow},
  D::AbstractVector{T},
  diag_Q::AbstractSparseVector{T},
  diagind_K,
  regu::Regularization{Tlow},
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
) where {T, Tlow}
  if (regu.regul == :dynamic || regu.regul == :hybrid) && K_fact isa LDLFactorizationData
    update_K_dynamic!(K, K_fact.LDL, regu, diagind_K, cnts, qp)
    @timeit_debug to "factorize" generic_factorize!(K, K_fact)
  elseif regu.regul == :classic
    @timeit_debug to "factorize" generic_factorize!(K, K_fact)
    while !RipQP.factorized(K_fact)
      out = update_regu_trycatch!(regu, cnts)
      out == 1 && return out
      cnts.c_catch += 1
      cnts.c_catch >= 4 && return 1
      update_K!(K, D, regu, s_l, s_u, x_m_lvar, uvar_m_x, ilow, iupp, diag_Q, diagind_K, nvar, ncon)
      @timeit_debug to "factorize" generic_factorize!(K, K_fact)
    end

  else # no Regularization
    @timeit_debug to "factorize" generic_factorize!(K, K_fact)
  end

  return 0 # factorization succeeded
end
