using .CUDA

function sparse_transpose_dropzeros(rows, cols, vals::CuVector, nrows, ncols)
  CPUvals = Vector(vals)
  MT = sparse(cols, rows, CPUvals, ncols, nrows)
  dropzeros!(MT)
  MTGPU = CUDA.CUSPARSE.CuSparseMatrixCSR(MT)
  return MTGPU
end

function get_diag_Q_dense(Q::CUDA.CUSPARSE.CuSparseMatrixCSR{T}) where {T <: Real}
  n = size(Q, 1)
  diagval = CUDA.zeros(T, n)
  fill_diag_Q_dense!(Q.rowPtr, Q.colVal, Q.nzVal, diagval, n)
  return diagval
end

function fill_diag_Q_dense!(
  Q_rowPtr,
  Q_colVal,
  Q_nzVal::CUDA.CuVector{T},
  diagval::CUDA.CuVector{T},
  n,
) where {T <: Real}
  function kernel(Q_rowPtr, Q_colVal, Q_nzVal, diagval, n)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    k = Q_rowPtr[index + 1] - 1
    if k > 0
      i = Q_colVal[k]
      if index == i
        diagval[index] = Q_nzVal[k]
      end
    end
    return nothing
  end

  threads = min(n, 256)
  blocks = ceil(Int, n / threads)
  @cuda name = "diagq" threads = threads blocks = blocks kernel(
    Q_rowPtr,
    Q_colVal,
    Q_nzVal,
    diagval,
    n,
  )
end

function check_bounds(x::T, lvar, uvar) where {T <: Real}
  ϵ = T(1.0e-4)
  if lvar >= x
    x = lvar + ϵ
  end
  if x >= uvar
    x = uvar - ϵ
  end
  if (lvar < x < uvar) == false
    x = (lvar + uvar) / 2
  end
  return x
end

# starting points
function update_rngbounds!(x, irng, lvar, uvar, ϵ) where {T <: Real}
  @views broadcast!(check_bounds, x[irng], x[irng], lvar[irng], uvar[irng])
end

# α computation (in iterations.jl)
function ddir(dir_vi::T, vi::T) where {T <: Real}
  if dir_vi < zero(T)
    α_new = -vi * T(0.999) / dir_vi
    return α_new
  end
  return one(T)
end

function compute_α_dual_gpu(v, dir_v, store_v)
  map!(ddir, store_v, dir_v, v)
  return minimum(store_v)
end

function pdir_l(dir_vi::T, lvari::T, vi::T) where {T <: Real}
  if dir_vi < zero(T)
    α_new = (lvari - vi) * T(0.999) / dir_vi
    return α_new
  end
  return one(T)
end

function pdir_u(dir_vi::T, uvari::T, vi::T) where {T <: Real}
  if dir_vi > zero(T)
    α_new = (uvari - vi) * T(0.999) / dir_vi
    return α_new
  end
  return one(T)
end

function compute_α_primal_gpu(v, dir_v, lvar, uvar, store_v)
  map!(pdir_l, store_v, dir_v, lvar, v)
  α_l = minimum(store_v)
  map!(pdir_u, store_v, dir_v, uvar, v)
  α_u = minimum(store_v)
  return min(α_l, α_u)
end

@inline function compute_αs_gpu(
  x::CuVector,
  s_l::CuVector,
  s_u::CuVector,
  lvar::CuVector,
  uvar::CuVector,
  Δxy::CuVector,
  Δs_l::CuVector,
  Δs_u::CuVector,
  nvar,
  store_vpri::CuVector,
  store_vdual_l::CuVector,
  store_vdual_u::CuVector,
)
  α_pri = @views compute_α_primal_gpu(x, Δxy[1:nvar], lvar, uvar, store_vpri)
  α_dual_l = compute_α_dual_gpu(s_l, Δs_l, store_vdual_l)
  α_dual_u = compute_α_dual_gpu(s_u, Δs_u, store_vdual_u)
  return α_pri, min(α_dual_l, α_dual_u)
end

# dot gpu with views 
function dual_obj_gpu(
  b,
  y,
  xTQx_2,
  s_l,
  s_u,
  lvar,
  uvar,
  c0,
  ilow,
  iupp,
  store_vdual_l,
  store_vdual_u,
)
  store_vdual_l .= @views lvar[ilow]
  store_vdual_u .= @views uvar[iupp]
  dual_obj = dot(b, y) - xTQx_2 + dot(s_l, store_vdual_l) - dot(s_u, store_vdual_u) + c0
  return dual_obj
end
