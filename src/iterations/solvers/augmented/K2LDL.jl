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
"""
mutable struct K2LDLParams{T, Fact} <: AugmentedParams
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
  ρ0::T = (T == Float16) ? one(T) : T(sqrt(eps()) * 1e5) ,
  δ0::T = (T == Float16) ? one(T) : T(sqrt(eps()) * 1e5) ,
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
  elseif iconf.mode == :multi
    regu = Regularization(T(sp.ρ0), T(sp.δ0), sqrt(eps(T)), sqrt(eps(T)), sp.fact_alg.regul)
  end
  diag_Q = get_diag_Q(fd.Q)
  K = Symmetric(create_K2(id, D, fd.Q.data, fd.A, diag_Q, regu), sp.uplo)

  diagind_K = get_diagind_K(K)
  K_fact = init_fact(K, sp.fact_alg)
  if regu.regul == :dynamic || regu.regul == :hybrid
    Amax = @views norm(K.data.nzval[diagind_K], Inf)
    # regu.ρ, regu.δ = T(eps(T)^(3 / 4)), T(eps(T)^(0.45))
    K_fact.r1, K_fact.r2 = -T(eps(T)^(3 / 4)), T(eps(T)^(0.45))
    K_fact.tol = Amax * T(eps(T))
    K_fact.n_d = id.nvar
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
  T0::DataType,
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
  T0::DataType,
) where {T <: Real}
  if (pad.regu.regul == :classic || pad.regu.regul == :hybrid) && cnts.k != 0
    # update ρ and δ values, check K diag magnitude 
    out = update_regu_diagK2!(pad.regu, pad.K, pad.diagind_K, id.nvar, itd, cnts, T, T0)
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
    T,
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
    T,
    T0,
  ) # update D and factorize K

  if out == 1
    pad.fact_fail = true
    return out
  end

  return 0
end

# Init functions for the K2 system
function fill_K2!(
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
  diag_Q_nzind,
  δ,
  ncon,
  nvar,
  regul,
)
  added_coeffs_diag = 0 # we add coefficients that do not appear in Q in position i,i if Q[i,i] = 0
  n_nz = length(diag_Q_nzind)
  c_nz = n_nz > 0 ? 1 : 0
  @inbounds for j = 1:nvar  # Q coeffs, tmp diag coefs. 
    K_colptr[j + 1] = Q_colptr[j + 1] + added_coeffs_diag
    for k = Q_colptr[j]:(Q_colptr[j + 1] - 2)
      nz_idx = k + added_coeffs_diag
      K_rowval[nz_idx] = Q_rowval[k]
      K_nzval[nz_idx] = -Q_nzval[k]
    end
    if c_nz == 0 || c_nz > n_nz || diag_Q_nzind[c_nz] != j
      added_coeffs_diag += 1
      K_colptr[j + 1] += 1
      nz_idx = K_colptr[j + 1] - 1
      K_rowval[nz_idx] = j
      K_nzval[nz_idx] = D[j]
    else
      if c_nz != 0
        c_nz += 1
      end
      nz_idx = K_colptr[j + 1] - 1
      K_rowval[nz_idx] = j
      K_nzval[nz_idx] = D[j] - Q_nzval[Q_colptr[j + 1] - 1]
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

function create_K2(
  id::QM_IntData,
  D::AbstractVector,
  Q::SparseMatrixCSC,
  A::SparseMatrixCSC,
  diag_Q::SparseVector,
  regu::Regularization;
  T = eltype(D),
)
  # for classic regul only
  n_nz = length(D) - length(diag_Q.nzind) + length(A.nzval) + length(Q.nzval)
  if (regu.regul == :classic || regu.regul == :hybrid) && regu.δ > 0
    n_nz += id.ncon
  end
  K_colptr = Vector{Int}(undef, id.ncon + id.nvar + 1)
  K_colptr[1] = 1
  K_rowval = Vector{Int}(undef, n_nz)
  K_nzval = Vector{T}(undef, n_nz)
  # [-Q -D    A]
  # [0       δI]

  fill_K2!(
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
    diag_Q.nzind,
    regu.δ,
    id.ncon,
    id.nvar,
    regu.regul,
  )

  return SparseMatrixCSC{T, Int}(id.ncon + id.nvar, id.ncon + id.nvar, K_colptr, K_rowval, K_nzval)
end

# function create_K2(
#   id::QM_IntData,
#   D::AbstractVector,
#   Q::SparseMatrixCOO,
#   A::SparseMatrixCOO,
#   diag_Q::SparseVector,
#   regu::Regularization;
#   T = eltype(D),
# )
#   block_row1 = hcat((Q + Diagonal(D)), coo_spzeros(T, id.nvar, id.ncon))
#   block_row1.vals .= .-block_row1.vals
#   block_row2 = hcat(A, regu.δ * I)
#   return vcat(block_row1, block_row2)
# end

function create_K2(
  id::QM_IntData,
  D::AbstractVector,
  Q::SparseMatrixCOO,
  A::SparseMatrixCOO,
  diag_Q::SparseVector,
  regu::Regularization;
  T = eltype(D),
)
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
  return SparseMatrixCOO(nvar + ncon, nvar + ncon, rows, cols, vals)
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
  K::Symmetric,
  D,
  regu,
  s_l,
  s_u,
  x_m_lvar,
  uvar_m_x,
  ilow,
  iupp,
  diag_Q,
  diagind_K,
  nvar,
  ncon,
  T,
)
  if regu.regul == :classic || regu.regul == :hybrid
    D .= -regu.ρ
    if regu.δ > zero(T)
      update_diag_K22!(K, regu.δ, diagind_K, nvar, ncon)
    end
  else
    D .= zero(T)
  end
  D[ilow] .-= s_l ./ x_m_lvar
  D[iupp] .-= s_u ./ uvar_m_x
  D[diag_Q.nzind] .-= diag_Q.nzval
  update_diag_K11!(K, D, diagind_K, nvar)
end

function update_K_dynamic!(K::Symmetric, K_fact, regu, diagind_K, cnts, T, qp)
  Amax = @views norm(K.data.nzval[diagind_K], Inf)
  if Amax > T(1e6) / K_fact.r2 && cnts.c_pdd < 8
    if T == Float32 && regu.regul == :dynamic
      return one(Int) # update to Float64
    elseif (qp || cnts.c_pdd < 4) && regu.regul == :dynamic
      cnts.c_pdd += 1
      regu.δ /= 10
      K_fact.r2 = regu.δ
    end
  end
  K_fact.tol = Amax * T(eps(T))
end

# iteration functions for the K2 system
function factorize_K2!(
  K::Symmetric,
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
  if regu.regul == :dynamic || regu.regul == :hybrid
    update_K_dynamic!(K, K_fact, regu, diagind_K, cnts, T, qp)
    generic_factorize!(K, K_fact)
  elseif regu.regul == :classic
    generic_factorize!(K, K_fact)
    while !factorized(K_fact)
      out = update_regu_trycatch!(regu, cnts, T, T0)
      out == 1 && return out
      cnts.c_catch += 1
      cnts.c_catch >= 4 && return 1
      update_K!(
        K,
        D,
        regu,
        s_l,
        s_u,
        x_m_lvar,
        uvar_m_x,
        ilow,
        iupp,
        diag_Q,
        diagind_K,
        nvar,
        ncon,
        T,
      )
      generic_factorize!(K, K_fact)
    end

  else # no Regularization
    generic_factorize!(K, K_fact)
  end

  return 0 # factorization succeeded
end

# conversion functions
function convertpad(
  ::Type{<:PreallocatedData{T}},
  pad::PreallocatedDataK2LDL{T_old},
  sp_old::K2LDLParams,
  sp_new::Union{Nothing, K2LDLParams},
  id::QM_IntData,
  fd::Abstract_QM_FloatData,
  T0::DataType,
) where {T <: Real, T_old <: Real}
  pad = PreallocatedDataK2LDL(
    convert(Array{T}, pad.D),
    convert(Regularization{T}, pad.regu),
    convert(SparseVector{T, Int}, pad.diag_Q),
    Symmetric(convert_mat(pad.K.data, T), Symbol(pad.K.uplo)),
    convertldl(T, pad.K_fact),
    pad.fact_fail,
    pad.diagind_K,
  )

  if pad.regu.regul == :classic
    if T == Float64 && typeof(sp_new) == Nothing
      pad.regu.ρ_min = 1e-5 * sqrt(eps())
      pad.regu.δ_min = 1e0 * sqrt(eps())
    else
      pad.regu.ρ_min, pad.regu.δ_min = T(sqrt(eps(T)) * 1e1), T(sqrt(eps(T)) * 1e1)
    end
  elseif pad.regu.regul == :dynamic
    pad.regu.ρ, pad.regu.δ = T(eps(T)^(3 / 4)), T(eps(T)^(0.45))
    pad.K_fact.r1, pad.K_fact.r2 = -pad.regu.ρ, pad.regu.δ
  end

  return pad
end
