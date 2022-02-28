return_one_if_zero(val::T) where {T <: Real} = (val == zero(T)) ? one(T) : val

function get_norm_rc_CSC!(v, A_colptr, A_rowval, A_nzval, n, ax)
  T = eltype(v)
  v .= zero(T)
  for j = 1:n
    @inbounds @simd for i = A_colptr[j]:(A_colptr[j + 1] - 1)
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
get_norm_rc!(v, A::SparseMatrixCSC, ax) = get_norm_rc_CSC!(v, A.colptr, A.rowval, A.nzval, size(A, 2), ax)

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

function mul_A_D1_D2_CSC!(A_colptr, A_rowval, A_nzval, d1, d2, r, c, uplo)
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
mul_A_D1_D2!(A::SparseMatrixCSC, d1, d2, R, C, uplo) = mul_A_D1_D2_CSC!(A.colptr, A.rowval, A.nzval, d1, d2, R.diag, C.diag, uplo)

function mul_A_D1_D2!(A, d1, d2, R, C, uplo)
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

function mul_A_D3_CSC!(A_colptr, A_rowval, A_nzval, n, d3, uplo)
  for j = 1:n
    @inbounds @simd for i = A_colptr[j]:(A_colptr[j + 1] - 1)
      A_nzval[i] *= (uplo == :U) ? d3[A_rowval[i]] : d3[j]
    end
  end
end
mul_A_D3!(A::SparseMatrixCSC, D3, uplo) = mul_A_D3_CSC!(A.colptr, A.rowval, A.nzval, size(A, 2), D3.diag, uplo) 

function mul_A_D3!(A, D3, uplo)
  if uplo == :U
    lmul!(D3, A)
  else
    rmul!(A, D3)
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

  get_norm_rc!(R_k.diag, A, :row)
  get_norm_rc!(C_k.diag, A, :col)
  convergence = maximum(abs.(one(T) .- R_k.diag)) <= ϵ && maximum(abs.(one(T) .- C_k.diag)) <= ϵ
  mul_A_D1_D2!(A, D1.diag, D2.diag, R_k, C_k, uplo)
  k = 1
  while !convergence && k < max_iter
    get_norm_rc!(R_k.diag, A, :row)
    get_norm_rc!(C_k.diag, A, :col)
    convergence = maximum(abs.(one(T) .- R_k.diag)) <= ϵ && maximum(abs.(one(T) .- C_k.diag)) <= ϵ
    mul_A_D1_D2!(A, D1.diag, D2.diag, R_k, C_k, uplo)
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

function scaling_Ruiz!(
  fd_T0::QM_FloatData{T},
  id::QM_IntData,
  sd::ScaleData{T},
  ϵ::T;
  max_iter::Int = 100,
) where {T <: Real}
  d1, d2, d3, r_k, c_k = sd.d1, sd.d2, sd.d3, sd.r_k, sd.c_k
  D1 = Diagonal(d1)
  D2 = Diagonal(d2)
  D3 = Diagonal(d3)
  R_k = Diagonal(r_k)
  C_k = Diagonal(c_k)
  # scaling Q (symmetric)
  if nnz(fd_T0.Q.data) > 0
    if fd_T0.uplo == :U
      equilibrate!(fd_T0.Q, D3, R_k; ϵ = ϵ, max_iter = max_iter)
    else
      equilibrate!(fd_T0.Q, D3, C_k; ϵ = ϵ, max_iter = max_iter)
    end
    mul_A_D3!(fd_T0.A, D3, fd_T0.uplo)
    fd_T0.c .*= d3
    fd_T0.lvar ./= d3
    fd_T0.uvar ./= d3
  end

  # r (resp. c) norm of rows of AT (resp. cols) 
  # scaling: D2 * AT * D1
  equilibrate!(fd_T0.A, D1, D2, R_k, C_k; ϵ = ϵ, max_iter = max_iter, uplo = fd_T0.uplo)

  nnz(fd_T0.Q.data) > 0 && mul_Q_D2!(fd_T0.Q.data, D2)
  fd_T0.b .*= d1
  fd_T0.c .*= d2
  fd_T0.lvar ./= d2
  fd_T0.uvar ./= d2
end

function div_D2D3_Q_D3D2_CSC!(Q_colptr, Q_rowval, Q_nzval, d2, d3, n)
  for j = 1:n
    @inbounds @simd for i = Q_colptr[j]:(Q_colptr[j + 1] - 1)
      Q_nzval[i] /= d2[Q_rowval[i]] * d2[j] * d3[Q_rowval[i]] * d3[j]
    end
  end
end
div_D2D3_Q_D3D2!(Q::SparseMatrixCSC, D2, D3) = div_D2D3_Q_D3D2_CSC!(Q.colptr, Q.rowval, Q.nzval, D2.diag, D3.diag, size(Q, 1))

function div_D2D3_Q_D3D2!(Q, D2, D3)
  ldiv!(D2, Q)
  ldiv!(D3, Q)
  rdiv!(Q, D2)
  rdiv!(Q, D3)
end

function div_D1_A_D2D3_CSC!(A_colptr, A_rowval, A_nzval, d1, d2, d3, n, uplo)
  for j = 1:n
    @inbounds @simd for i = A_colptr[j]:(A_colptr[j + 1] - 1)
      A_nzval[i] /=
        (uplo == :U) ? d1[j] * d2[A_rowval[i]] * d3[A_rowval[i]] : d1[A_rowval[i]] * d2[j] * d3[j]
    end
  end
end
div_D1_A_D2D3!(A::SparseMatrixCSC, D1, D2, D3, uplo) = div_D1_A_D2D3_CSC!(A.colptr, A.rowval, A.nzval, D1.diag, D2.diag, D3.diag, size(A, 2), uplo)

function div_D1_A_D2D3!(A, D1, D2, D3, uplo)
  if uplo == :U
    rdiv!(A, D1)
    ldiv!(D2, A)
    ldiv!(D3, A)
  else
    ldiv!(D1, A)
    rdiv!(A, D2)
    rdiv!(A, D3)
  end
end

function post_scale!(
  d1::AbstractVector{T},
  d2::AbstractVector{T},
  d3::AbstractVector{T},
  pt::Point{T},
  res::AbstractResiduals{T},
  fd_T0::QM_FloatData{T},
  id::QM_IntData,
  itd::IterData{T},
) where {T <: Real}
  D1, D2, D3 = Diagonal(d1), Diagonal(d2), Diagonal(d3)
  pt.x .*= d2 .* d3
  nnz(fd_T0.Q.data) > 0 && div_D2D3_Q_D3D2!(fd_T0.Q.data, D2, D3)
  mul!(itd.Qx, fd_T0.Q, pt.x)
  itd.xTQx_2 = dot(pt.x, itd.Qx) / 2
  # div_D1_A_D2D3!(fd_T0.A.colptr, fd_T0.A.rowval, fd_T0.A.nzval, d1, d2, d3, fd_T0.A.n, fd_T0.uplo)
  div_D1_A_D2D3!(fd_T0.A, D1, D2, D3, fd_T0.uplo)
  pt.y .*= d1
  fd_T0.uplo == :U ? mul!(itd.ATy, fd_T0.A, pt.y) : mul!(itd.ATy, fd_T0.A', pt.y)
  fd_T0.uplo == :U ? mul!(itd.Ax, fd_T0.A', pt.x) : mul!(itd.Ax, fd_T0.A, pt.x)
  fd_T0.b ./= d1
  fd_T0.c ./= d2 .* d3
  itd.cTx = dot(fd_T0.c, pt.x)
  itd.pri_obj = itd.xTQx_2 + itd.cTx + fd_T0.c0
  fd_T0.lvar .*= d2 .* d3
  fd_T0.uvar .*= d2 .* d3
  pt.s_l ./= @views d2[id.ilow] .* d3[id.ilow]
  pt.s_u ./= @views d2[id.iupp] .* d3[id.iupp]
  itd.dual_obj =
    dot(fd_T0.b, pt.y) - itd.xTQx_2 + dot(pt.s_l, view(fd_T0.lvar, id.ilow)) -
    dot(pt.s_u, view(fd_T0.uvar, id.iupp)) + fd_T0.c0
  res.rb .= itd.Ax .- fd_T0.b
  res.rc .= itd.ATy .- itd.Qx .- fd_T0.c
  res.rc[id.ilow] .+= pt.s_l
  res.rc[id.iupp] .-= pt.s_u
  #         rcNorm, rbNorm = norm(rc), norm(rb)
  res.rcNorm, res.rbNorm = norm(res.rc, Inf), norm(res.rb, Inf)
end
