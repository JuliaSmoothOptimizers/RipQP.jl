function get_norm_rc!(v, AT_colptr, AT_rowval, AT_nzval, n, ax)
  T = eltype(v)
  v .= zero(T)
  for j = 1:n
    @inbounds @simd for i = AT_colptr[j]:(AT_colptr[j + 1] - 1)
      k = ax == :row ? AT_rowval[i] : j
      if abs(AT_nzval[i]) > v[k]
        v[k] = abs(AT_nzval[i])
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

function mul_AT_D1_D2!(AT_colptr, AT_rowval, AT_nzval, d1, d2, r, c, uplo)
  for j = 1:length(c)
    @inbounds @simd for i = AT_colptr[j]:(AT_colptr[j + 1] - 1)
      AT_nzval[i] /= r[AT_rowval[i]] * c[j]
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

function mul_AT_D3!(AT_colptr, AT_rowval, AT_nzval, n, d3, uplo)
  for j = 1:n
    @inbounds @simd for i = AT_colptr[j]:(AT_colptr[j + 1] - 1)
      AT_nzval[i] *= (uplo == :U) ? d3[AT_rowval[i]] : d3[j]
    end
  end
end

function mul_Q_D!(Q_colptr, Q_rowval, Q_nzval, d, c)
  for j = 1:length(d)
    @inbounds @simd for i = Q_colptr[j]:(Q_colptr[j + 1] - 1)
      Q_nzval[i] /= c[Q_rowval[i]] * c[j]
    end
  end
  d ./= c
end

function mul_Q_D2!(Q_colptr, Q_rowval, Q_nzval, d2)
  for j = 1:length(d2)
    @inbounds @simd for i = Q_colptr[j]:(Q_colptr[j + 1] - 1)
      Q_nzval[i] *= d2[Q_rowval[i]] * d2[j]
    end
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
  # scaling Q (symmetric)
  if length(fd_T0.Q.rowval) > 0
    if fd_T0.uplo == :U
      get_norm_rc!(r_k, fd_T0.Q.colptr, fd_T0.Q.rowval, fd_T0.Q.nzval, id.nvar, :row)
      convergence = maximum(abs.(one(T) .- r_k)) <= ϵ
      mul_Q_D!(fd_T0.Q.colptr, fd_T0.Q.rowval, fd_T0.Q.nzval, d3, r_k)
    else
      get_norm_rc!(c_k, fd_T0.Q.colptr, fd_T0.Q.rowval, fd_T0.Q.nzval, id.nvar, :col)
      convergence = maximum(abs.(one(T) .- c_k)) <= ϵ
      mul_Q_D!(fd_T0.Q.colptr, fd_T0.Q.rowval, fd_T0.Q.nzval, d3, c_k)
    end
    k = 1
    while !convergence && k < max_iter
      if fd_T0.uplo == :U
        get_norm_rc!(r_k, fd_T0.Q.colptr, fd_T0.Q.rowval, fd_T0.Q.nzval, id.nvar, :row)
        convergence = maximum(abs.(one(T) .- r_k)) <= ϵ
        mul_Q_D!(fd_T0.Q.colptr, fd_T0.Q.rowval, fd_T0.Q.nzval, d3, r_k)
      else
        get_norm_rc!(c_k, fd_T0.Q.colptr, fd_T0.Q.rowval, fd_T0.Q.nzval, id.nvar, :col)
        convergence = maximum(abs.(one(T) .- c_k)) <= ϵ
        mul_Q_D!(fd_T0.Q.colptr, fd_T0.Q.rowval, fd_T0.Q.nzval, d3, c_k)
      end
      k += 1
    end
    mul_AT_D3!(fd_T0.A.colptr, fd_T0.A.rowval, fd_T0.A.nzval, fd_T0.A.n, d3, fd_T0.uplo)
    fd_T0.c .*= d3
    fd_T0.lvar ./= d3
    fd_T0.uvar ./= d3
  end 

  # r (resp. c) norm of rows of AT (resp. cols) 
  # scaling: D2 * AT * D1
  get_norm_rc!(r_k, fd_T0.A.colptr, fd_T0.A.rowval, fd_T0.A.nzval, fd_T0.A.n, :row)
  get_norm_rc!(c_k, fd_T0.A.colptr, fd_T0.A.rowval, fd_T0.A.nzval, fd_T0.A.n, :col)
  convergence = maximum(abs.(one(T) .- r_k)) <= ϵ && maximum(abs.(one(T) .- c_k)) <= ϵ
  mul_AT_D1_D2!(fd_T0.A.colptr, fd_T0.A.rowval, fd_T0.A.nzval, d1, d2, r_k, c_k, fd_T0.uplo)
  k = 1
  while !convergence && k < max_iter
    get_norm_rc!(r_k, fd_T0.A.colptr, fd_T0.A.rowval, fd_T0.A.nzval, fd_T0.A.n, :row)
    get_norm_rc!(c_k, fd_T0.A.colptr, fd_T0.A.rowval, fd_T0.A.nzval, fd_T0.A.n, :col)
    convergence = maximum(abs.(one(T) .- r_k)) <= ϵ && maximum(abs.(one(T) .- c_k)) <= ϵ
    mul_AT_D1_D2!(fd_T0.A.colptr, fd_T0.A.rowval, fd_T0.A.nzval, d1, d2, r_k, c_k, fd_T0.uplo)
    k += 1
  end
  length(fd_T0.Q.rowval) > 0 && mul_Q_D2!(fd_T0.Q.colptr, fd_T0.Q.rowval, fd_T0.Q.nzval, d2)
  fd_T0.b .*= d1
  fd_T0.c .*= d2
  fd_T0.lvar ./= d2
  fd_T0.uvar ./= d2
end

function div_D2D3_Q_D3D2!(Q_colptr, Q_rowval, Q_nzval, d2, d3, n)
  for j = 1:n
    @inbounds @simd for i = Q_colptr[j]:(Q_colptr[j + 1] - 1)
      Q_nzval[i] /= d2[Q_rowval[i]] * d2[j] * d3[Q_rowval[i]] * d3[j]
    end
  end
end

function div_D1_A_D2D3!(AT_colptr, AT_rowval, AT_nzval, d1, d2, d3, n, uplo)
  for j = 1:n
    @inbounds @simd for i = AT_colptr[j]:(AT_colptr[j + 1] - 1)
      AT_nzval[i] /= (uplo == :U) ? d1[j] * d2[AT_rowval[i]] * d3[AT_rowval[i]] : d1[AT_rowval[i]] * d2[j] * d3[j]
    end
  end
end

function post_scale!(
  d1::Vector{T},
  d2::Vector{T},
  d3::Vector{T},
  pt::Point{T},
  res::AbstractResiduals{T},
  fd_T0::QM_FloatData{T},
  id::QM_IntData,
  itd::IterData{T},
) where {T <: Real}
  pt.x .*= d2 .* d3
  div_D2D3_Q_D3D2!(fd_T0.Q.colptr, fd_T0.Q.rowval, fd_T0.Q.nzval, d2, d3, id.nvar)
  mul!(itd.Qx, Symmetric(fd_T0.Q, fd_T0.uplo), pt.x)
  itd.xTQx_2 = dot(pt.x, itd.Qx) / 2
  div_D1_A_D2D3!(fd_T0.A.colptr, fd_T0.A.rowval, fd_T0.A.nzval, d1, d2, d3, fd_T0.A.n, fd_T0.uplo)
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
