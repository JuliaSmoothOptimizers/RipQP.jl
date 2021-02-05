function get_QM_data(QM :: QuadraticModel)
    T = eltype(QM.meta.lvar)
    # constructs A and Q transposed so we can create J_augm upper triangular. 
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

function convert_FloatData(T :: DataType, fd_T0 :: QM_FloatData{T0}) where {T0<:Real}
    return QM_FloatData(SparseMatrixCSC{T, Int}(fd_T0.Q.m, fd_T0.Q.n, 
                                                fd_T0.Q.colptr, fd_T0.Q.rowval, Array{T}(fd_T0.Q.nzval)),
                        SparseMatrixCSC{T, Int}(fd_T0.AT.m, fd_T0.AT.n, 
                                                fd_T0.AT.colptr, fd_T0.AT.rowval, Array{T}(fd_T0.AT.nzval)),
                        Array{T}(fd_T0.b), 
                        Array{T}(fd_T0.c), 
                        T(fd_T0.c0),
                        Array{T}(fd_T0.lvar), 
                        Array{T}(fd_T0.uvar))
end

function init_params(fd_T :: QM_FloatData{T}, id :: QM_IntData, ϵ :: tolerances{T}, sc :: stop_crit{Tc},
                     regul :: Symbol, mode :: Symbol, Rfunc :: RipQP_func) where {T<:Real, Tc<:Real}

    res = residuals(zeros(T, id.n_rows), zeros(T, id.n_cols), zero(T), zero(T), zero(T))
    
    itd = Rfunc.create_iterdata(fd_T, id, mode, regul)

    pad = preallocated_data(zeros(T, id.n_cols+id.n_rows+id.n_low+id.n_upp), # Δ_aff
                            zeros(T, id.n_cols+id.n_rows+id.n_low+id.n_upp), # Δ_cc
                            zeros(T, id.n_cols+id.n_rows+id.n_low+id.n_upp), # Δ
                            zeros(T, id.n_cols+id.n_rows), # Δ_xy
                            zeros(T, id.n_low), # x_m_l_αΔ_aff
                            zeros(T, id.n_upp), # u_m_x_αΔ_aff
                            zeros(T, id.n_low), # s_l_αΔ_aff
                            zeros(T, id.n_upp), # s_u_αΔ_aff
                            zeros(T, id.n_low), # rxs_l
                            zeros(T, id.n_upp) #rxs_u
                            )

    pt, itd, pad.Δ_xy = @views starting_points(fd_T, id, itd, pad.Δ_xy)

    # stopping criterion
    #     rcNorm, rbNorm = norm(rc), norm(rb)
    #     optimal = pdd < ϵ_pdd && rbNorm < ϵ_rb && rcNorm < ϵ_rc
    res.rb .= itd.Ax .- fd_T.b
    res.rc .= itd.ATy .-itd.Qx .+ pt.s_l .- pt.s_u .- fd_T.c
    res.rcNorm, res.rbNorm = norm(res.rc, Inf), norm(res.rb, Inf)
    ϵ.tol_rb, ϵ.tol_rc = ϵ.rb*(one(T) + res.rbNorm), ϵ.rc*(one(T) + res.rcNorm)

    sc.optimal = itd.pdd < ϵ.pdd && res.rbNorm < ϵ.tol_rb && res.rcNorm < ϵ.tol_rc
    sc.small_μ = itd.μ < ϵ.μ

    return itd, ϵ, pad, pt, res, sc
end

function convert_types!(T :: DataType, pt :: point{T_old}, itd :: iter_data{T_old}, res :: residuals{T_old},
                        pad :: preallocated_data{T_old}, T0 :: DataType) where {T_old<:Real}

   pt = convert(point{T}, pt)
   res = convert(residuals{T}, res)
   itd = convert(iter_data{T}, itd)
   if T == Float64 && T0 == Float64
       itd.regu.ρ_min, itd.regu.δ_min = T(sqrt(eps())*1e-5), T(sqrt(eps())*1e0)
   else
       itd.regu.ρ_min, itd.regu.δ_min = T(sqrt(eps(T))*1e1), T(sqrt(eps(T))*1e1)
   end
   pad = convert(preallocated_data{T}, pad)

   itd.regu.ρ /= 10
   itd.regu.δ /= 10

   return pt, itd, res, pad
end

############ tools for sparse matrices ##############

function get_diag_sparseCSC(M_colptr :: Vector{Int}, n :: Int; tri :: Symbol =:U)
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

function get_diag_Q(Q_colptr :: Vector{Int}, Q_rowval :: Vector{Int}, Q_nzval :: Vector{T}, 
                    n :: Int) where {T<:Real} 
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