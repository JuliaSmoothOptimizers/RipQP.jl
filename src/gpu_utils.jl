using .CUDA

include("iterations/solvers/gpu/K2KrylovLDLGPU.jl")

function get_mat_QPData(
  A::CUDA.CUSPARSE.CuSparseMatrixCOO{T, Ti},
  H::CUDA.CUSPARSE.CuSparseMatrixCOO{T, Ti},
  nvar::Int,
  ncon::Int,
  sp::K2KrylovGPUParams,
) where {T, Ti}
  # A is Aᵀ of QuadraticModel QM
  fdA = CUDA.CUSPARSE.CuSparseMatrixCSC(
    sparse(Vector(A.colInd), Vector(A.rowInd), Vector(A.nzVal), nvar, ncon),
  )
  fdQ = CUDA.CUSPARSE.CuSparseMatrixCSC(
    sparse(Vector(H.colInd), Vector(H.rowInd), Vector(H.nzVal), nvar, nvar),
  )
  return fdA, Symmetric(fdQ, sp.uplo)
end

change_vector_eltype(S0::Type{<:CUDA.CuVector}, ::Type{T}) where {T} =
  S0.name.wrapper{T, 1, CUDA.Mem.DeviceBuffer}

convert_mat(M::CUDA.CUSPARSE.CuSparseMatrixCSC, ::Type{T}) where {T} =
  CUDA.CUSPARSE.CuSparseMatrixCSC(
    convert(CUDA.CuArray{Int, 1, CUDA.Mem.DeviceBuffer}, M.colPtr),
    convert(CUDA.CuArray{Int, 1, CUDA.Mem.DeviceBuffer}, M.rowVal),
    convert(CUDA.CuArray{T, 1, CUDA.Mem.DeviceBuffer}, M.nzVal),
    M.dims,
  )
convert_mat(M::CUDA.CUSPARSE.CuSparseMatrixCSR, ::Type{T}) where {T} =
  CUDA.CUSPARSE.CuSparseMatrixCSR(
    convert(CUDA.CuArray{Int, 1, CUDA.Mem.DeviceBuffer}, M.rowPtr),
    convert(CUDA.CuArray{Int, 1, CUDA.Mem.DeviceBuffer}, M.colVal),
    convert(CUDA.CuArray{T, 1, CUDA.Mem.DeviceBuffer}, M.nzVal),
    M.dims,
  )
convert_mat(M::CUDA.CuMatrix, ::Type{T}) where {T} =
  convert(typeof(M).name.wrapper{T, 2, CUDA.Mem.DeviceBuffer}, M)

function sparse_dropzeros(rows, cols, vals::CuVector, nrows, ncols)
  CPUvals = Vector(vals)
  M = sparse(rows, cols, CPUvals, ncols, nrows)
  dropzeros!(M)
  MGPU = CUDA.CUSPARSE.CuSparseMatrixCSR(M)
  return MGPU
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
function update_rngbounds!(x, irng, lvar, uvar, ϵ)
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

import LinearAlgebra.rdiv!, LinearAlgebra.ldiv!

function LinearAlgebra.rdiv!(M::CUDA.AbstractGPUArray{T}, D::Diagonal) where {T}
  D.diag .= one(T) ./ D.diag
  rmul!(M, D)
  D.diag .= one(T) ./ D.diag
  return M
end

function LinearAlgebra.ldiv!(D::Diagonal, M::CUDA.AbstractGPUArray{T}) where {T}
  D.diag .= one(T) ./ D.diag
  lmul!(D, M)
  D.diag .= one(T) ./ D.diag
  return M
end

import NLPModelsModifiers.SlackModel

function slackdata2(
  data::QuadraticModels.QPData{T, S, M1, M2},
  meta::NLPModelsModifiers.NLPModelMeta{T},
  ns::Int,
) where {T, S, M1 <: CUDA.CUSPARSE.CuSparseMatrixCOO, M2 <: CUDA.CUSPARSE.CuSparseMatrixCOO}
  nvar_slack = meta.nvar + ns
  Ti = (T == Float64) ? Int64 : Int32
  return QuadraticModels.QPData(
    copy(data.c0),
    [data.c; fill!(similar(data.c, ns), zero(T))],
    CUDA.CUSPARSE.CuSparseMatrixCOO{T, Ti}(
      Ti.(data.H.rowInd),
      Ti.(data.H.colInd),
      data.H.nzVal,
      (nvar_slack, nvar_slack),
      length(data.H.nzVal),
    ),
    CUDA.CUSPARSE.CuSparseMatrixCOO{T, Ti}(
      CuVector{Ti}([Vector(data.A.rowInd); meta.jlow; meta.jupp; meta.jrng]),
      CuVector{Ti}([Vector(data.A.colInd); (meta.nvar + 1):(meta.nvar + ns)]),
      CuVector([Vector(data.A.nzVal); fill!(Vector{T}(undef, ns), -one(T))]),
      (meta.ncon, nvar_slack),
      length(data.A.nzVal),
    ),
  )
end

function NLPModelsModifiers.SlackModel(
  qp::AbstractQuadraticModel{T, S},
  name = qp.meta.name * "-slack",
) where {T, S <: CUDA.CuArray}
  qp.meta.ncon == length(qp.meta.jfix) && return qp
  nfix = length(qp.meta.jfix)
  ns = qp.meta.ncon - nfix

  data = slackdata2(qp.data, qp.meta, ns)

  meta = NLPModelsModifiers.slack_meta(qp.meta, name = qp.meta.name)

  return QuadraticModel(meta, NLPModelsModifiers.Counters(), data)
end
