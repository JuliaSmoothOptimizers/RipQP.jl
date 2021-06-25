function vcatsort(v1, v2)
  n2 = length(v2)
  n2 == 0 && return v1
  n1 = length(v1)
  n1 == 0 && return v2

  n = n1 + n2
  res = similar(v1, n)
  c1, c2 = 1, 1
  @inbounds for i = 1:n
    if c2 == n2 + 1
      res[i] = v1[c1]
      c1 += 1
    elseif c1 == n1 + 1
      res[i] = v2[c2]
      c2 += 1
    else
      if v1[c1] < v2[c2]
        res[i] = v1[c1]
        c1 += 1
      else
        res[i] = v2[c2]
        c2 += 1
      end
    end
  end

  return res
end

function sparse_transpose_dropzeros(rows, cols, vals::Vector, nrows, ncols)
  MT = sparse(cols, rows, vals, ncols, nrows)
  dropzeros!(MT)
  return MT
end

function get_QM_data(QM::QuadraticModel)
  # constructs A and Q transposed so we can create K upper triangular. 
  # As Q is symmetric (but lower triangular in QuadraticModels.jl) we leave its name unchanged.
  AT = sparse_transpose_dropzeros(
    QM.data.Arows,
    QM.data.Acols,
    QM.data.Avals,
    QM.meta.ncon,
    QM.meta.nvar,
  )
  Q = sparse_transpose_dropzeros(
    QM.data.Hrows,
    QM.data.Hcols,
    QM.data.Hvals,
    QM.meta.nvar,
    QM.meta.nvar,
  )
  id = QM_IntData(
    vcatsort(QM.meta.ilow, QM.meta.irng),
    vcatsort(QM.meta.iupp, QM.meta.irng),
    QM.meta.irng,
    QM.meta.ifree,
    QM.meta.ncon,
    QM.meta.nvar,
    0,
    0,
  )
  id.nlow, id.nupp = length(id.ilow), length(id.iupp) # number of finite constraints
  @assert QM.meta.lcon == QM.meta.ucon # equality constraint (Ax=b)
  fd = QM_FloatData(Q, AT, QM.meta.lcon, QM.data.c, QM.data.c0, QM.meta.lvar, QM.meta.uvar)
  return fd, id
end

function allocate_workspace(QM::QuadraticModel, iconf::InputConfig, itol::InputTol, start_time, T0::DataType)

  sc = StopCrit(false, false, false, itol.max_iter, itol.max_time, start_time, 0.0)

  # save inital IntData to compute multipliers at the end of the algorithm
  idi = IntDataInit(
    QM.meta.nvar,
    QM.meta.ncon,
    QM.meta.ilow,
    QM.meta.iupp,
    QM.meta.irng,
    QM.meta.ifix,
    QM.meta.jlow,
    QM.meta.jupp,
    QM.meta.jrng,
    QM.meta.jfix,
  )

  SlackModel!(QM) # add slack variables to the problem if QM.meta.lcon != QM.meta.ucon

  fd_T0, id = get_QM_data(QM)

  T = T0 # T0 is the data type, in mode :multi T will gradually increase to T0
  ϵ = Tolerances(
    T(itol.ϵ_pdd),
    T(itol.ϵ_rb),
    T(itol.ϵ_rc),
    one(T),
    one(T),
    T(itol.ϵ_μ),
    T(itol.ϵ_Δx),
    iconf.normalize_rtol,
  )

  S0 = typeof(fd_T0.c)
  if iconf.scaling
    sd = ScaleData{T0, S0}(
      fill!(S0(undef, id.ncon), one(T0)),
      fill!(S0(undef, id.nvar), one(T0)),
      fill!(S0(undef, id.nvar), one(T0)),
      S0(undef, id.nvar),
      S0(undef, id.ncon),
    )
    scaling_Ruiz!(fd_T0, id, sd, T(1.0e-3))
  else
    empty_v = S0(undef, 0)
    sd = ScaleData{T0, S0}(empty_v, empty_v, empty_v, empty_v, empty_v)
  end

  if iconf.mode == :multi
    T = Float32
    ϵ32 = Tolerances(
      T(itol.ϵ_pdd32),
      T(itol.ϵ_rb32),
      T(itol.ϵ_rc32),
      one(T),
      one(T),
      T(itol.ϵ_μ),
      T(itol.ϵ_Δx),
      iconf.normalize_rtol,
    )
    fd32 = convert_FloatData(T, fd_T0)
    res, itd, dda = allocate_iter_workspace_T(fd32, id, ϵ32, iconf, T0)
    if T0 == Float64
      return sc, idi, fd_T0, id, ϵ, res, itd, dda, sd, T, ϵ32, fd32
    elseif T0 == Float128
      T = Float64
      fd64 = convert_FloatData(T, fd_T0)
      ϵ64 = Tolerances(
        T(itol.ϵ_pdd64),
        T(itol.ϵ_rb64),
        T(itol.ϵ_rc64),
        one(T),
        one(T),
        T(itol.ϵ_μ),
        T(itol.ϵ_Δx),
        iconf.normalize_rtol,
      )
      T = Float32
      return sc, idi, fd_T0, id, ϵ, res, itd, dda, sd, T, ϵ32, fd32, ϵ64, fd64
    end
  elseif iconf.mode == :mono
    res, itd, dda = allocate_iter_workspace_T(fd_T0, id, ϵ, iconf, T0) 
    return sc, idi, fd_T0, id, ϵ, res, itd, dda, sd, T
  end
end

function allocate_iter_workspace_T(fd::QM_FloatData{T}, id::QM_IntData, ϵ::Tolerances{T}, iconf::InputConfig, 
                                   T0::DataType) where {T <: Real}

  res = Residuals(similar(fd.c, id.ncon), similar(fd.c, id.nvar), zero(T), zero(T))

  itd = IterData(
    similar(fd.c, id.nvar + id.ncon), # Δxy
    similar(fd.c, id.nlow), # Δs_l
    similar(fd.c, id.nupp), # Δs_u
    similar(fd.c, id.nlow), # x_m_lvar
    similar(fd.c, id.nupp), # uvar_m_x
    similar(fd.c, id.nvar), # init Qx
    similar(fd.c, id.nvar), # init ATy
    similar(fd.c, id.ncon), # Ax
    zero(T), #xTQx
    zero(T), #cTx
    zero(T), #pri_obj
    zero(T), #dual_obj
    zero(T), #μ
    zero(T),#pdd
    zeros(T, 6), #l_pdd
    one(T), #mean_pdd
    nnz(fd.Q) > 0,
  )

  dda_type = Symbol(:DescentDirectionAllocs, iconf.solve_method)
  dda = eval(dda_type)(id, fd)

  return res, itd, dda
end

function initialize(
  fd::QM_FloatData{T},
  id::QM_IntData,
  res::Residuals{T},
  itd::IterData{T},
  dda::DescentDirectionAllocs{T},
  iconf::InputConfig{Tconf},
  T0::DataType,
) where {T <: Real, Tconf <: Real}

  pad = PreallocatedData(iconf.sp, fd, id, iconf)

  # init system
  # solve [-Q-D    A' ] [x] = [0]  to initialize (x, y, s_l, s_u)
  #       [  A     0  ] [y] = [b]
  itd.Δxy[1:(id.nvar)] .= 0
  itd.Δxy[(id.nvar + 1):end] = fd.b

  cnts = Counters(zero(Int), zero(Int), 0, 0, iconf.kc, iconf.max_ref, zero(Int), iconf.w)

  pt0 = Point(
    similar(fd.c, id.nvar),
    similar(fd.c, id.ncon),
    similar(fd.c, id.nlow),
    similar(fd.c, id.nupp),
  )
  out = solver!(pad, dda, pt0, itd, fd, id, res, cnts, T0, :init)
  pt0.x .= itd.Δxy[1:(id.nvar)]
  pt0.y .= itd.Δxy[(id.nvar + 1):end]

  return pad, pt0, cnts
end

function init_params(
  fd_T::QM_FloatData{T},
  id::QM_IntData,
  res::Residuals{T},
  itd::IterData{T},
  dda::DescentDirectionAllocs{T},
  ϵ::Tolerances{T},
  sc::StopCrit{Tc},
  iconf::InputConfig{Tconf},
  T0::DataType,
) where {T <: Real, Tc <: Real, Tconf <: Real}

  pad, pt, cnts = initialize(fd_T, id, res, itd, dda, iconf, T0)

  starting_points!(pt, fd_T, id, itd)

  # stopping criterion
  #     rcNorm, rbNorm = norm(rc), norm(rb)
  #     optimal = pdd < ϵ_pdd && rbNorm < ϵ_rb && rcNorm < ϵ_rc
  res.rb .= itd.Ax .- fd_T.b
  res.rc .= itd.ATy .- itd.Qx .- fd_T.c
  res.rc[id.ilow] .+= pt.s_l
  res.rc[id.iupp] .-= pt.s_u
  res.rcNorm, res.rbNorm = norm(res.rc, Inf), norm(res.rb, Inf)
  set_tol_residuals!(ϵ, res.rbNorm, res.rcNorm)

  sc.optimal = itd.pdd < ϵ.pdd && res.rbNorm < ϵ.tol_rb && res.rcNorm < ϵ.tol_rc
  sc.small_μ = itd.μ < ϵ.μ

  return itd, ϵ, pad, pt, sc, cnts
end

function set_tol_residuals!(ϵ::Tolerances{T}, rbNorm::T, rcNorm::T) where {T <: Real}
  if ϵ.normalize_rtol == true
    ϵ.tol_rb, ϵ.tol_rc = ϵ.rb * (one(T) + rbNorm), ϵ.rc * (one(T) + rcNorm)
  else
    ϵ.tol_rb, ϵ.tol_rc = ϵ.rb, ϵ.rc
  end
end

############ tools for sparse matrices ##############

function get_diag_sparseCSC(M_colptr, n; tri = :U)
  # get diagonal index of M.nzval
  # we assume all columns of M are non empty, and M triangular (:L or :U)
  @assert tri == :U || tri == :L
  if tri == :U
    diagind = M_colptr[2:end] .- one(Int)
  else
    diagind = M_colptr[1:(end - 1)]
  end
  return diagind
end

function get_diag_Q(Q_colptr, Q_rowval, Q_nzval::Vector{T}, n) where {T <: Real}
  diagval = spzeros(T, n)
  @inbounds @simd for j = 1:n
    for k = Q_colptr[j]:(Q_colptr[j + 1] - 1)
      if j == Q_rowval[k]
        diagval[j] = Q_nzval[k]
      end
    end
  end
  return diagval
end
