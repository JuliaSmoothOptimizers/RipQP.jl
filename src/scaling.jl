return_one_if_zero(val::T) where {T <: Real} = (val == zero(T)) ? one(T) : val

function get_norm_rc_CSC!(v, A_colptr, A_rowval, A_nzval, n, ax)
  T = eltype(v)
  v .= zero(T)
  for j = 1:n
    @inbounds for i = A_colptr[j]:(A_colptr[j + 1] - 1)
      k = ax == :row ? A_rowval[i] : j
      if abs(A_nzval[i]) > v[k]
        v[k] = abs(A_nzval[i])
      end
    end
  end

  v .= sqrt.(v)
  @inbounds @simd for i = 1:length(v)
    if v[i] == zero(T)
      v[i] = one(T)
    end
  end
end
get_norm_rc!(v, A::SparseMatrixCSC, ax) =
  get_norm_rc_CSC!(v, A.colptr, A.rowval, A.nzval, size(A, 2), ax)

function get_norm_rc!(v, A, ax)
  T = eltype(v)
  v .= zero(T)
  if ax == :row
    maximum!(abs, v, A)
  elseif ax == :col
    maximum!(abs, v', A)
  end
  v .= return_one_if_zero.(sqrt.(v))
end

function mul_D1_A_D2_CSC!(A_colptr, A_rowval, A_nzval, d1, d2, r, c, uplo)
  for j = 1:length(c)
    @inbounds @simd for i = A_colptr[j]:(A_colptr[j + 1] - 1)
      A_nzval[i] /= r[A_rowval[i]] * c[j]
    end
  end

  if uplo == :U
    d1 ./= c
    d2 ./= r
  else
    d1 ./= r
    d2 ./= c
  end
end
mul_D1_A_D2(A::SparseMatrixCSC, d1, d2, R, C, uplo) =
  mul_D1_A_D2_CSC!(A.colptr, A.rowval, A.nzval, d1, d2, R.diag, C.diag, uplo)

function mul_D1_A_D2(A, d1, d2, R, C, uplo)
  ldiv!(R, A)
  rdiv!(A, C)
  if uplo == :U
    d1 ./= C.diag
    d2 ./= R.diag
  else
    d1 ./= R.diag
    d2 ./= C.diag
  end
end

function mul_Q_D_CSC!(Q_colptr, Q_rowval, Q_nzval, d, c)
  for j = 1:length(d)
    @inbounds @simd for i = Q_colptr[j]:(Q_colptr[j + 1] - 1)
      Q_nzval[i] /= c[Q_rowval[i]] * c[j]
    end
  end
  d ./= c
end
mul_Q_D!(Q::SparseMatrixCSC, d, C) = mul_Q_D_CSC!(Q.colptr, Q.rowval, Q.nzval, d, C.diag)

function mul_Q_D!(Q, d, C)
  ldiv!(C, Q)
  rdiv!(Q, C)
  d ./= C.diag
end

function mul_Q_D2_CSC!(Q_colptr, Q_rowval, Q_nzval, d2)
  for j = 1:length(d2)
    @inbounds @simd for i = Q_colptr[j]:(Q_colptr[j + 1] - 1)
      Q_nzval[i] *= d2[Q_rowval[i]] * d2[j]
    end
  end
end
mul_Q_D2!(Q::SparseMatrixCSC, D2) = mul_Q_D2_CSC!(Q.colptr, Q.rowval, Q.nzval, D2.diag)

function mul_Q_D2!(Q, D2)
  lmul!(D2, Q)
  rmul!(Q, D2)
end

# equilibration scaling (transform A so that its rows and cols have an infinite norm close to 1): 
# A ← D2 * A * D1 (uplo = :L), R_k, C_k are storage Diagonal Arrays that have the same size as D1, D2
# or A ← D2 * Aᵀ * D1 (uplo = :U), R_k, C_k are storage Diagonal Arrays that have the same size as D2, D1
# ϵ is the norm tolerance on the row and cols infinite norm of A
# max_iter is the maximum number of iterations
function equilibrate!(
  A::AbstractMatrix{T},
  D1::Diagonal{T, S},
  D2::Diagonal{T, S},
  R_k::Diagonal{T, S},
  C_k::Diagonal{T, S};
  ϵ::T = T(1.0e-2),
  max_iter::Int = 100,
  uplo::Symbol = :L,
) where {T <: Real, S <: AbstractVector{T}}
  min(size(A)...) == 0 && return
  get_norm_rc!(R_k.diag, A, :row)
  get_norm_rc!(C_k.diag, A, :col)
  convergence = maximum(abs.(one(T) .- R_k.diag)) <= ϵ && maximum(abs.(one(T) .- C_k.diag)) <= ϵ
  mul_D1_A_D2(A, D1.diag, D2.diag, R_k, C_k, uplo)
  k = 1
  while !convergence && k < max_iter
    get_norm_rc!(R_k.diag, A, :row)
    get_norm_rc!(C_k.diag, A, :col)
    convergence = maximum(abs.(one(T) .- R_k.diag)) <= ϵ && maximum(abs.(one(T) .- C_k.diag)) <= ϵ
    mul_D1_A_D2(A, D1.diag, D2.diag, R_k, C_k, uplo)
    k += 1
  end
end

# Symmetric equilibration scaling (transform Q so that its rows and cols have an infinite norm close to 1): 
# Q ← D3 * Q * D3, C_k is a storage Diagonal Array that has the same size as D3
# ϵ is the norm tolerance on the row and cols infinite norm of A
# max_iter is the maximum number of iterations
function equilibrate!(
  Q::Symmetric{T},
  D3::Diagonal{T, S},
  C_k::Diagonal{T, S};
  ϵ::T = T(1.0e-2),
  max_iter::Int = 100,
) where {T <: Real, S <: AbstractVector{T}}
  size(Q, 1) == 0 && return
  get_norm_rc!(C_k.diag, Q.data, :col)
  convergence = maximum(abs.(one(T) .- C_k.diag)) <= ϵ
  mul_Q_D!(Q.data, D3.diag, C_k)
  k = 1
  while !convergence && k < max_iter
    get_norm_rc!(C_k.diag, Q, :col)
    convergence = maximum(abs.(one(T) .- C_k.diag)) <= ϵ
    mul_Q_D!(Q.data, D3.diag, C_k)
    k += 1
  end
end

function scaling!(
  fd_T0::QM_FloatData{T},
  id::QM_IntData,
  sd::ScaleDataLP{T},
  ϵ::T;
  max_iter::Int = 100,
) where {T <: Real}
  d1, d2, r_k, c_k = sd.d1, sd.d2, sd.r_k, sd.c_k
  d1 .= one(T)
  d2 .= one(T)
  D1 = Diagonal(d1)
  D2 = Diagonal(d2)
  R_k = Diagonal(r_k)
  C_k = Diagonal(c_k)

  # r (resp. c) norm of rows of AT (resp. cols) 
  # scaling: D2 * AT * D1
  equilibrate!(fd_T0.A, D2, D1, C_k, R_k; ϵ = ϵ, max_iter = max_iter, uplo = fd_T0.uplo)

  fd_T0.b .*= d2
  fd_T0.c .*= d1
  fd_T0.lvar ./= d1
  fd_T0.uvar ./= d1
end

function div_D_Q_D_CSC!(Q_colptr, Q_rowval, Q_nzval, d, n)
  for j = 1:n
    @inbounds @simd for i = Q_colptr[j]:(Q_colptr[j + 1] - 1)
      Q_nzval[i] /= d[Q_rowval[i]] * d[j]
    end
  end
end
div_D_Q_D!(Q::SparseMatrixCSC, D) = div_D_Q_D_CSC!(Q.colptr, Q.rowval, Q.nzval, D.diag, size(Q, 1))

function div_D_Q_D!(Q, D)
  ldiv!(D, Q)
  rdiv!(Q, D)
end

function div_D1_A_D2_CSC!(A_colptr, A_rowval, A_nzval, d1, d2, n, uplo)
  for j = 1:n
    @inbounds @simd for i = A_colptr[j]:(A_colptr[j + 1] - 1)
      A_nzval[i] /= (uplo == :U) ? d1[j] * d2[A_rowval[i]] : d1[A_rowval[i]] * d2[j]
    end
  end
end
div_D1_A_D2!(A::SparseMatrixCSC, D1, D2, uplo) =
  div_D1_A_D2_CSC!(A.colptr, A.rowval, A.nzval, D1.diag, D2.diag, size(A, 2), uplo)

function div_D1_A_D2!(A, D1, D2, uplo)
  if uplo == :U
    rdiv!(A, D1)
    ldiv!(D2, A)
  else
    ldiv!(D1, A)
    rdiv!(A, D2)
  end
end

function post_scale!(
  sd::ScaleData{T},
  pt::Point{T},
  res::AbstractResiduals{T},
  fd_T0::QM_FloatData{T},
  id::QM_IntData,
  itd::IterData{T},
) where {T <: Real}
  if typeof(sd) <: ScaleDataLP
    d1 = sd.d1
    d2 = sd.d2
  elseif typeof(sd) <: ScaleDataQP
    d1 = view(sd.deq, 1:(id.nvar))
    d2 = view(sd.deq, (id.nvar + 1):(id.nvar + id.ncon))
  end
  D1, D2 = Diagonal(d1), Diagonal(d2)

  # unscale problem data
  nnz(fd_T0.Q.data) > 0 && div_D_Q_D!(fd_T0.Q.data, D1)
  div_D1_A_D2!(fd_T0.A, D2, D1, fd_T0.uplo)
  fd_T0.b ./= d2
  fd_T0.c ./= d1
  fd_T0.lvar .*= d1
  fd_T0.uvar .*= d1

  # update final point
  pt.x .*= d1
  pt.y .*= d2
  pt.s_l ./= @views d1[id.ilow]
  pt.s_u ./= @views d1[id.iupp]

  # update iter data
  mul!(itd.Qx, fd_T0.Q, pt.x)
  itd.xTQx_2 = dot(pt.x, itd.Qx) / 2
  fd_T0.uplo == :U ? mul!(itd.ATy, fd_T0.A, pt.y) : mul!(itd.ATy, fd_T0.A', pt.y)
  fd_T0.uplo == :U ? mul!(itd.Ax, fd_T0.A', pt.x) : mul!(itd.Ax, fd_T0.A, pt.x)
  itd.cTx = dot(fd_T0.c, pt.x)
  itd.pri_obj = itd.xTQx_2 + itd.cTx + fd_T0.c0
  itd.dual_obj =
    dot(fd_T0.b, pt.y) - itd.xTQx_2 + dot(pt.s_l, view(fd_T0.lvar, id.ilow)) -
    dot(pt.s_u, view(fd_T0.uvar, id.iupp)) + fd_T0.c0

  # update residuals
  res.rb .= itd.Ax .- fd_T0.b
  res.rc .= itd.ATy .- itd.Qx .- fd_T0.c
  res.rc[id.ilow] .+= pt.s_l
  res.rc[id.iupp] .-= pt.s_u
  #         rcNorm, rbNorm = norm(rc), norm(rb)
  res.rcNorm, res.rbNorm = norm(res.rc, Inf), norm(res.rb, Inf)
end

# equilibrate K2
function get_norm_rc_K2_CSC!(
  v,
  Q_colptr,
  Q_rowval,
  Q_nzval,
  A_colptr,
  A_rowval,
  A_nzval,
  D,
  deq,
  δ,
  nvar,
  ncon,
  uplo,
)
  T = eltype(v)
  v .= zero(T)
  ignore_D = length(D) == 0 # ignore D if it is empty (in case we want to scale [Q Aᵀ; A δI])
  for j = 1:nvar
    passed_diagj = false
    @inbounds for k = Q_colptr[j]:(Q_colptr[j + 1] - 1)
      i = Q_rowval[k]
      if i == j
        nzvalij = ignore_D ? -deq[i]^2 * Q_nzval[k] : deq[i]^2 * (-Q_nzval[k] + D[j])
        passed_diagj = true
      else
        nzvalij = -deq[i] * Q_nzval[k] * deq[j]
      end
      if abs(nzvalij) > v[i]
        v[i] = abs(nzvalij)
      end
      if abs(nzvalij) > v[j]
        v[j] = abs(nzvalij)
      end
    end
    if !passed_diagj
      nzvalij = ignore_D ? zero(T) : deq[j]^2 * D[j]
      if abs(nzvalij) > v[j]
        v[j] = abs(nzvalij)
      end
    end
  end

  ncolA = (uplo == :L) ? nvar : ncon
  for j = 1:ncolA
    @inbounds for k = A_colptr[j]:(A_colptr[j + 1] - 1)
      if uplo == :L
        iup = A_rowval[k] + nvar
        jup = j
      else
        iup = A_rowval[k]
        jup = j + nvar
      end
      nzvalij = deq[iup] * A_nzval[k] * deq[jup]
      if abs(nzvalij) > v[iup]
        v[iup] = abs(nzvalij)
      end
      if abs(nzvalij) > v[jup]
        v[jup] = abs(nzvalij)
      end
    end
  end

  if δ > 0
    @inbounds for j = (nvar + 1):(nvar + ncon)
      rdij = δ * deq[j]^2
      if abs(rdij) > v[j]
        v[j] = abs(rdij)
      end
    end
  end

  v .= sqrt.(v)
  @inbounds @simd for i = 1:length(v)
    if v[i] == zero(T)
      v[i] = one(T)
    end
  end
end

get_norm_rc_K2!(v, Q::SparseMatrixCSC, A::SparseMatrixCSC, D, deq, δ, nvar, ncon, uplo) =
  get_norm_rc_K2_CSC!(
    v,
    Q.colptr,
    Q.rowval,
    Q.nzval,
    A.colptr,
    A.rowval,
    A.nzval,
    D,
    deq,
    δ,
    nvar,
    ncon,
    uplo,
  )

get_norm_rc_K2!(
  v,
  Q::Symmetric{T, SparseMatrixCSC{T, Int}},
  A::SparseMatrixCSC,
  D,
  deq,
  δ,
  nvar,
  ncon,
  uplo,
) where {T} = get_norm_rc_K2!(v, Q.data, A, D, deq, δ, nvar, ncon, uplo)

# not efficient but can be improved:
function get_norm_rc_K2!(v, Q::Symmetric, A, D, deq, δ, nvar, ncon, uplo)
  # D as storage vec
  @assert δ == 0
  T = eltype(v)
  v .= zero(T)
  v1 = view(v, 1:nvar)
  v2 = view(v, (nvar + 1):(nvar + ncon))
  Deq1 = Diagonal(view(deq, 1:nvar))
  Deq2 = Diagonal(view(deq, (nvar + 1):(nvar + ncon)))
  rmul!(Q.data, Deq1)
  lmul!(Deq1, Q.data)
  maximum!(abs, v1, Q)
  rdiv!(Q.data, Deq1)
  ldiv!(Deq1, Q.data)
  if uplo == :U
    lmul!(Deq1, A)
    rmul!(A, Deq2)
    maximum!(abs, D, A)
    maximum!(abs, v2', A)
    ldiv!(Deq1, A)
    rdiv!(A, Deq2)
  else
    lmul!(Deq2, A)
    rmul!(A, Deq1)
    maximum!(abs, v2, A)
    maximum!(abs, D', A)
    ldiv!(Deq2, A)
    rdiv!(A, Deq1)
  end
  v1 .= max.(v1, D)
  v .= return_one_if_zero.(sqrt.(v))
end

function equilibrate_K2!(
  Q::SparseMatrixCSC{T},
  A::SparseMatrixCSC{T},
  D::Vector{T},
  δ::T,
  nvar::Int,
  ncon::Int,
  Deq::Diagonal{T, S},
  C_k::Diagonal{T, S},
  uplo::Symbol;
  ϵ::T = T(1.0e-2),
  max_iter::Int = 100,
) where {T <: Real, S <: AbstractVector{T}}
  Deq.diag .= one(T)
  # get_norm_rc!(C_k.diag, Q.data, :col)
  get_norm_rc_K2!(C_k.diag, Q, A, D, Deq.diag, δ, nvar, ncon, uplo)
  convergence = maximum(abs.(one(T) .- C_k.diag)) <= ϵ
  Deq.diag ./= C_k.diag
  k = 1
  while !convergence && k < max_iter
    # get_norm_rc!(C_k.diag, Q, :col)
    get_norm_rc_K2!(C_k.diag, Q, A, D, Deq.diag, δ, nvar, ncon, uplo)
    convergence = maximum(abs.(one(T) .- C_k.diag)) <= ϵ
    Deq.diag ./= C_k.diag
    k += 1
  end
end

function scaling!(
  fd_T0::QM_FloatData{T},
  id::QM_IntData,
  sd::ScaleDataQP{T},
  ϵ::T;
  max_iter::Int = 100,
) where {T <: Real}
  size(fd_T0.Q) == 0 && min(size(fd_T0.A)...) == 0 && return
  deq, c_k = sd.deq, sd.c_k
  deq .= one(T)
  Deq = Diagonal(deq)
  C_k = Diagonal(c_k)
  # empty vector to use ignore_D in get_norm_rc_K2 or storage vector if not csc
  Dtmp = typeof(fd_T0.A) <: SparseMatrixCSC ? similar(deq, 0) : similar(deq, id.nvar)
  δ = zero(T)
  # scaling Q (symmetric)
  get_norm_rc_K2!(C_k.diag, fd_T0.Q, fd_T0.A, Dtmp, Deq.diag, δ, id.nvar, id.ncon, fd_T0.uplo)
  convergence = maximum(abs.(one(T) .- C_k.diag)) <= ϵ
  Deq.diag ./= C_k.diag
  k = 1
  while !convergence && k < max_iter
    get_norm_rc_K2!(C_k.diag, fd_T0.Q, fd_T0.A, Dtmp, Deq.diag, δ, id.nvar, id.ncon, fd_T0.uplo)
    convergence = maximum(abs.(one(T) .- C_k.diag)) <= ϵ
    Deq.diag ./= C_k.diag
    k += 1
  end

  D1 = Diagonal(view(deq, 1:(id.nvar)))
  D2 = Diagonal(view(deq, (id.nvar + 1):(id.nvar + id.ncon)))
  if fd_T0.uplo == :U
    lmul!(D1, fd_T0.A)
    rmul!(fd_T0.A, D2)
  else
    lmul!(D2, fd_T0.A)
    rmul!(fd_T0.A, D1)
  end
  mul_Q_D2!(fd_T0.Q.data, D1)
  fd_T0.c .*= D1.diag
  fd_T0.lvar ./= D1.diag
  fd_T0.uvar ./= D1.diag
  fd_T0.b .*= D2.diag
end
