function get_QM_data(QM :: QuadraticModel)
    T = eltype(QM.meta.lvar)
    # constructs A and Q transposed so we can create K upper triangular. 
    # As Q is symmetric (but lower triangular in QuadraticModels.jl) we leave its name unchanged.
    AT = sparse(QM.data.Acols, QM.data.Arows, QM.data.Avals, QM.meta.nvar, QM.meta.ncon) 
    dropzeros!(AT)
    Q = sparse(QM.data.Hcols, QM.data.Hrows, QM.data.Hvals, QM.meta.nvar, QM.meta.nvar)  
    dropzeros!(Q)
    id = QM_IntData([QM.meta.ilow; QM.meta.irng], [QM.meta.iupp; QM.meta.irng], QM.meta.irng, 
                    QM.meta.ncon, QM.meta.nvar, 0, 0)
    id.n_low, id.n_upp = length(id.ilow), length(id.iupp) # number of finite constraints
    @assert QM.meta.lcon == QM.meta.ucon # equality constraint (Ax=b)
    fd_T = QM_FloatData(Q, AT, QM.meta.lcon, QM.data.c, QM.data.c0, QM.meta.lvar, QM.meta.uvar)
    return fd_T, id, T
end

function init_params(fd_T :: QM_FloatData{T}, id :: QM_IntData, ϵ :: tolerances{T}, sc :: stop_crit{Tc},
                     regul :: Symbol, mode :: Symbol, create_iterdata :: Function) where {T<:Real, Tc<:Real}

    res = residuals(zeros(T, id.n_rows), zeros(T, id.n_cols), zero(T), zero(T), zero(T))
    
    itd, pad, pt = create_iterdata(fd_T, id, mode, regul)

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

    return itd, ϵ, pad, pt, res, sc
end

function set_tol_residuals!(ϵ :: tolerances{T}, rbNorm :: T, rcNorm :: T) where {T<:Real}
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