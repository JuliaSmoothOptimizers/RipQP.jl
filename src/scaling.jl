function get_norm_rc!(v, AT_colptr, AT_rowval, AT_nzval, n, ax)
  T = eltype(v)
  v .= zero(T)
  @inbounds @simd for j = 1:n
    for i = AT_colptr[j]:(AT_colptr[j + 1] - 1)
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

function get_norm_rk_AT!(v, AT_colptr, AT_rowval, AT_nzval, n, d3)
  T = eltype(v)
  v .= zero(T)
  @inbounds @simd for j = 1:n
    for i = AT_colptr[j]:(AT_colptr[j + 1] - 1)
      k = AT_rowval[i]
      if d3[k] == one(T) && abs(AT_nzval[i]) > v[k]
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

function mul_AT_D1_D2!(AT_colptr, AT_rowval, AT_nzval, d1, d2, r, c)
  @inbounds @simd for j = 1:length(c)
    for i = AT_colptr[j]:(AT_colptr[j + 1] - 1)
      AT_nzval[i] /= r[AT_rowval[i]] * c[j]
    end
  end
  d1 ./= c
  d2 ./= r
end

function mul_AT_D3!(AT_colptr, AT_rowval, AT_nzval, n, d3)
  @inbounds @simd for j = 1:n
    for i = AT_colptr[j]:(AT_colptr[j + 1] - 1)
      AT_nzval[i] *= d3[AT_rowval[i]]
    end
  end
end

function mul_Q_D!(Q_colptr, Q_rowval, Q_nzval, d, c)
  @inbounds @simd for j = 1:length(d)
    for i = Q_colptr[j]:(Q_colptr[j + 1] - 1)
      Q_nzval[i] /= c[Q_rowval[i]] * c[j]
    end
  end
  d ./= c
end

function mul_Q_D2!(Q_colptr, Q_rowval, Q_nzval, d2)
  @inbounds @simd for j = 1:length(d2)
    for i = Q_colptr[j]:(Q_colptr[j + 1] - 1)
      Q_nzval[i] *= d2[Q_rowval[i]] * d2[j]
    end
  end
end

function scaling_Ruiz!(
  fd_T0::QM_FloatData{T},
  id::QM_IntData,
  ϵ::T;
  max_iter::Int = 100,
) where {T <: Real}

	r_k, c_k = zeros(T, id.nvar), zeros(T, id.ncon)
  # scaling Q (symmetric)
  d3 = ones(T, id.nvar)
  if length(fd_T0.Q.rowval) > 0
    # r_k .= zero(T) # r_k is now norm of rows of Q
    get_norm_rc!(r_k, fd_T0.Q.colptr, fd_T0.Q.rowval, fd_T0.Q.nzval, id.nvar, :row)
    convergence = maximum(abs.(one(T) .- r_k)) <= ϵ
    mul_Q_D!(fd_T0.Q.colptr, fd_T0.Q.rowval, fd_T0.Q.nzval, d3, r_k)
    k = 1
    while !convergence && k < max_iter
      get_norm_rc!(r_k, fd_T0.Q.colptr, fd_T0.Q.rowval, fd_T0.Q.nzval, id.nvar, :row)
      convergence = maximum(abs.(one(T) .- r_k)) <= ϵ
      mul_Q_D!(fd_T0.Q.colptr, fd_T0.Q.rowval, fd_T0.Q.nzval, d3, r_k)
      k += 1
    end

    mul_AT_D3!(fd_T0.AT.colptr, fd_T0.AT.rowval, fd_T0.AT.nzval, fd_T0.AT.n, d3)
    fd_T0.c .*= d3
    fd_T0.lvar ./= d3
    fd_T0.uvar ./= d3
  end 

  d1, d2 = ones(T, id.ncon), ones(T, id.nvar)
  # r (resp. c) norm of rows of AT (resp. cols) 
  # scaling: D2 * AT * D1
	r_k .= zero(T)
  get_norm_rk_AT!(r_k, fd_T0.AT.colptr, fd_T0.AT.rowval, fd_T0.AT.nzval, id.ncon, d3)
  get_norm_rc!(c_k, fd_T0.AT.colptr, fd_T0.AT.rowval, fd_T0.AT.nzval, id.ncon, :col)
  convergence = maximum(abs.(one(T) .- c_k)) <= ϵ
  mul_AT_D1_D2!(fd_T0.AT.colptr, fd_T0.AT.rowval, fd_T0.AT.nzval, d1, d2, r_k, c_k)
  k = 1
  while !convergence && k < max_iter
    get_norm_rk_AT!(r_k, fd_T0.AT.colptr, fd_T0.AT.rowval, fd_T0.AT.nzval, id.ncon, d3)
    get_norm_rc!(c_k, fd_T0.AT.colptr, fd_T0.AT.rowval, fd_T0.AT.nzval, id.ncon, :col)
    convergence = maximum(abs.(one(T) .- c_k)) <= ϵ
    mul_AT_D1_D2!(fd_T0.AT.colptr, fd_T0.AT.rowval, fd_T0.AT.nzval, d1, d2, r_k, c_k)
    k += 1
  end

  # mul_Q_D2!(fd_T0.Q.colptr, fd_T0.Q.rowval, fd_T0.Q.nzval, d2)
  fd_T0.b .*= d1
  fd_T0.c .*= d2
  fd_T0.lvar ./= d2
  fd_T0.uvar ./= d2

  return fd_T0, d1, d2, d3
end

function div_D2D3_Q_D3D2!(Q_colptr, Q_rowval, Q_nzval, d2, d3, n)
  @inbounds @simd for j = 1:n
    for i = Q_colptr[j]:(Q_colptr[j + 1] - 1)
      Q_nzval[i] /= d2[Q_rowval[i]] * d2[j] * d3[Q_rowval[i]] * d3[j]
    end
  end
end

function div_D1_A_D2D3!(AT_colptr, AT_rowval, AT_nzval, d1, d2, d3, n)
  @inbounds @simd for j = 1:n
    for i = AT_colptr[j]:(AT_colptr[j + 1] - 1)
      AT_nzval[i] /= d1[j] * d2[AT_rowval[i]] * d3[AT_rowval[i]]
    end
  end
end

function post_scale(
  d1::Vector{T},
  d2::Vector{T},
  d3::Vector{T},
  pt::Point{T},
  res::Residuals{T},
  fd_T0::QM_FloatData{T},
  id::QM_IntData,
  Qx::Vector{T},
  ATy::Vector{T},
  Ax::Vector{T},
  cTx::T,
  pri_obj::T,
  dual_obj::T,
  xTQx_2::T,
) where {T <: Real}
  pt.x .*= d2 .* d3
  div_D2D3_Q_D3D2!(fd_T0.Q.colptr, fd_T0.Q.rowval, fd_T0.Q.nzval, d2, d3, id.nvar)
  Qx = mul!(Qx, Symmetric(fd_T0.Q, :U), pt.x)
  xTQx_2 = dot(pt.x, Qx) / 2
  div_D1_A_D2D3!(fd_T0.AT.colptr, fd_T0.AT.rowval, fd_T0.AT.nzval, d1, d2, d3, id.ncon)
  pt.y .*= d1
  ATy = mul!(ATy, fd_T0.AT, pt.y)
  Ax = mul!(Ax, fd_T0.AT', pt.x)
  fd_T0.b ./= d1
  fd_T0.c ./= d2 .* d3
  cTx = dot(fd_T0.c, pt.x)
  pri_obj = xTQx_2 + cTx + fd_T0.c0
  fd_T0.lvar .*= d2 .* d3
  fd_T0.uvar .*= d2 .* d3
  pt.s_l ./= @views d2[id.ilow] .* d3[id.ilow]
  pt.s_u ./= @views d2[id.iupp] .* d3[id.iupp]
  dual_obj =
    dot(fd_T0.b, pt.y) - xTQx_2 + dot(pt.s_l, view(fd_T0.lvar, id.ilow)) -
    dot(pt.s_u, view(fd_T0.uvar, id.iupp)) + fd_T0.c0
  res.rb .= Ax .- fd_T0.b
  res.rc .= ATy .- Qx .- fd_T0.c
  res.rc[id.ilow] .+= pt.s_l
  res.rc[id.iupp] .-= pt.s_u
  #         rcNorm, rbNorm = norm(rc), norm(rb)
  res.rcNorm, res.rbNorm = norm(res.rc, Inf), norm(res.rb, Inf)

  return pt, pri_obj, res
end
