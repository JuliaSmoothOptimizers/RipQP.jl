function get_QM_data(QM :: QuadraticModel)
    T = eltype(QM.meta.lvar)
    # constructs A and Q transposed so we can create K upper triangular. 
    # As Q is symmetric (but lower triangular in QuadraticModels.jl) we leave its name unchanged.
    AT = sparse(QM.data.Acols, QM.data.Arows, QM.data.Avals, QM.meta.nvar, QM.meta.ncon) 
    dropzeros!(AT)
    Q = sparse(QM.data.Hcols, QM.data.Hrows, QM.data.Hvals, QM.meta.nvar, QM.meta.nvar)  
    dropzeros!(Q)
    id = QM_IntData([QM.meta.ilow; QM.meta.irng], [QM.meta.iupp; QM.meta.irng], QM.meta.irng, QM.meta.ifree,
                    QM.meta.ncon, QM.meta.nvar, 0, 0)
    id.nlow, id.nupp = length(id.ilow), length(id.iupp) # number of finite constraints
    @assert QM.meta.lcon == QM.meta.ucon # equality constraint (Ax=b)
    fd_T = QM_FloatData(Q, AT, QM.meta.lcon, QM.data.c, QM.data.c0, QM.meta.lvar, QM.meta.uvar)
    return fd_T, id, T
end

function initialize(fd :: QM_FloatData{T}, id :: QM_IntData, res :: Residuals{T}, iconf :: InputConfig{Tconf}, 
                    T0 :: DataType) where {T<:Real, Tconf<:Real}

    itd = IterData(zeros(T, id.nvar+id.ncon), # Δxy
                   zeros(T, id.nlow), # Δs_l
                   zeros(T, id.nupp), # Δs_u
                   zeros(T, id.nlow), # x_m_lvar
                   zeros(T, id.nupp), # uvar_m_x
                   zeros(T, id.nvar), # init Qx
                   zeros(T, id.nvar), # init ATy
                   zeros(T, id.ncon), # Ax
                   zero(T), #xTQx
                   zero(T), #cTx
                   zero(T), #pri_obj
                   zero(T), #dual_obj
                   zero(T), #μ
                   zero(T),#pdd
                   zeros(T, 6), #l_pdd
                   one(T), #mean_pdd
                   nnz(fd.Q) > 0
                   )
    
    dda_type = Symbol(:DescentDirectionAllocs, iconf.solve_method)
    dda = eval(dda_type)(id, T)
    
    pad = PreallocatedData(iconf.sp, fd, id, iconf)
    
    # init system
    # solve [-Q-D    A' ] [x] = [b]  to initialize (x, y, s_l, s_u)
    #       [  A     0  ] [y] = [0]
    itd.Δxy[id.nvar+1: end] = fd.b

    cnts = Counters(zero(Int), zero(Int), 0, 0, iconf.kc, iconf.max_ref, zero(Int))

    pt0 = Point(zeros(T, id.nvar), zeros(T, id.ncon), zeros(T, id.nlow), zeros(T, id.nupp))
    out = solver!(pad, dda, pt0, itd, fd, id, res, cnts, T0, :init)
    pt0.x .= itd.Δxy[1:id.nvar]
    pt0.y .= itd.Δxy[id.nvar+1:end]

    return itd, dda, pad, pt0, cnts
end

function init_params(fd_T :: QM_FloatData{T}, id :: QM_IntData, ϵ :: Tolerances{T}, sc :: StopCrit{Tc},
                     iconf :: InputConfig{Tconf}, T0 :: DataType) where {T<:Real, Tc<:Real, Tconf<:Real}

    res = Residuals(zeros(T, id.ncon), zeros(T, id.nvar), zero(T), zero(T), zero(T))
    
    itd, dda, pad, pt, cnts = initialize(fd_T, id, res, iconf, T0)

    starting_points!(pt, fd_T, id, itd)

    # stopping criterion
    #     rcNorm, rbNorm = norm(rc), norm(rb)
    #     optimal = pdd < ϵ_pdd && rbNorm < ϵ_rb && rcNorm < ϵ_rc
    res.rb .= itd.Ax .- fd_T.b
    res.rc .= itd.ATy .-itd.Qx .- fd_T.c
    res.rc[id.ilow] .+= pt.s_l
    res.rc[id.iupp] .-= pt.s_u
    res.rcNorm, res.rbNorm = norm(res.rc, Inf), norm(res.rb, Inf)
    set_tol_residuals!(ϵ, res.rbNorm, res.rcNorm)

    sc.optimal = itd.pdd < ϵ.pdd && res.rbNorm < ϵ.tol_rb && res.rcNorm < ϵ.tol_rc
    sc.small_μ = itd.μ < ϵ.μ

    return itd, ϵ, dda, pad, pt, res, sc, cnts
end

function set_tol_residuals!(ϵ :: Tolerances{T}, rbNorm :: T, rcNorm :: T) where {T<:Real}
    if ϵ.normalize_rtol == true
        ϵ.tol_rb, ϵ.tol_rc = ϵ.rb*(one(T) + rbNorm), ϵ.rc*(one(T) + rcNorm)
    else
        ϵ.tol_rb, ϵ.tol_rc = ϵ.rb, ϵ.rc
    end
end

############ tools for sparse matrices ##############

function get_diag_sparseCSC(M_colptr, n; tri =:U)
    # get diagonal index of M.nzval
    # we assume all columns of M are non empty, and M triangular (:L or :U)
    @assert tri ==:U || tri == :L
    diagind = zeros(Int, n) # square matrix
    if tri == :U
        @inbounds @simd for i=1:n
            diagind[i] = M_colptr[i+1] - 1
        end
    else
        @inbounds @simd for i=1:n
            diagind[i] = M_colptr[i]
        end
    end
    return diagind
end

function get_diag_Q(Q_colptr, Q_rowval, Q_nzval :: Vector{T}, n) where {T<:Real} 
    diagval = spzeros(T, n)
    @inbounds @simd for j=1:n
        for k=Q_colptr[j]:(Q_colptr[j+1]-1)
            if j == Q_rowval[k]
                diagval[j] = Q_nzval[k]
            end
        end
    end
    return diagval
end