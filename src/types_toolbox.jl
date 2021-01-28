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

function init_params(fd_T0 :: QM_FloatData{T0}, id :: QM_IntData,
                     ϵ_T :: tolerances{T}, ϵ :: tolerances{T0}, regul :: Symbol) where{T<:Real, T0<:Real}

    fd_T = convert_FloatData(T, fd_T0)
    res = residuals(zeros(T, id.n_rows), zeros(T, id.n_cols), zero(T), zero(T), zero(T))
    # init regularization values
    regu = regularization(T(sqrt(eps())*1e5), T(sqrt(eps())*1e5), T(sqrt(eps(T))*1e0), T(sqrt(eps(T))*1e0), regul)
    tmp_diag = -T(1.0e-2) .* ones(T, id.n_cols)
    diag_Q = get_diag_Q(fd_T.Q)
    J_augm = create_J_augm(id, tmp_diag, fd_T.Q, fd_T.AT, diag_Q, regu, T)
    diagind_J = get_diag_sparseCSC(J_augm)
    J_fact = ldl_analyze(Symmetric(J_augm, :U))
    if regu.regul == :dynamic
        Amax = @views norm(J_augm.nzval[diagind_J], Inf)
        J_fact = ldl_factorize!(Symmetric(J_augm, :U), J_fact, tol=Amax*T(eps(T)), r1=T(-eps(T)^(3/4)),
                                r2=T(sqrt(eps(T))), n_d=id.n_cols)
    else
        J_fact = ldl_factorize!(Symmetric(J_augm, :U), J_fact)
    end
    J_fact.__factorized = true
    itd = iter_data(tmp_diag, # tmp diag
                    diag_Q, #diag_Q
                    J_augm, #J_augm
                    J_fact, #J_fact
                    diagind_J, #diagind_J
                    zeros(T, id.n_low), # x_m_lvar
                    zeros(T, id.n_upp), # uvar_m_x
                    zeros(T, id.n_cols), # init Qx
                    zeros(T, id.n_cols), # init ATy
                    zeros(T, id.n_rows), # Ax
                    zero(T), #xTQx
                    zero(T), #cTx
                    zero(T), #pri_obj
                    zero(T), #dual_obj
                    zero(T), #μ
                    zero(T),#pdd
                    zeros(T, 6), #l_pdd
                    one(T) #mean_pdd
                    )

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
    ϵ_T.tol_rb, ϵ_T.tol_rc = ϵ_T.rb*(one(T) + res.rbNorm), ϵ_T.rc*(one(T) + res.rcNorm)
    ϵ.tol_rb, ϵ.tol_rc = ϵ.rb*(one(T0) + T0(res.rbNorm)), ϵ.rc*(one(T0) + T0(res.rcNorm))
    sc = stop_crit(itd.pdd < ϵ_T.pdd && res.rbNorm < ϵ_T.tol_rb && res.rcNorm < ϵ_T.tol_rc, # optimal
                   false, # small_Δx
                   itd.μ < ϵ_T.μ, # small_μ
                   false # tired
                   )

    return fd_T, ϵ_T, ϵ, regu, itd, pad, pt, res, sc
end

function init_params_mono(fd_T :: QM_FloatData{T}, id :: QM_IntData, ϵ :: tolerances{T},
                          regul :: Symbol) where {T<:Real}

    res = residuals(zeros(T, id.n_rows), zeros(T, id.n_cols), zero(T), zero(T), zero(T))
    # init regularization values
    regu = regularization(T(sqrt(eps())*1e5), T(sqrt(eps())*1e5), 1e-5*sqrt(eps(T)), 1e0*sqrt(eps(T)), regul)
    tmp_diag = -T(1.0e0)/2 .* ones(T, id.n_cols)
    diag_Q = get_diag_Q(fd_T.Q)
    J_augm = create_J_augm(id, tmp_diag, fd_T.Q, fd_T.AT, diag_Q, regu, T)
    diagind_J = get_diag_sparseCSC(J_augm)
    J_fact = ldl_analyze(Symmetric(J_augm, :U))
    if regu.regul == :dynamic
        Amax = @views norm(J_augm.nzval[diagind_J], Inf)
        J_fact = ldl_factorize!(Symmetric(J_augm, :U), J_fact, tol=Amax*T(eps(T)), r1=T(-eps(T)^(3/4)),
                                r2=T(sqrt(eps(T))), n_d=id.n_cols)
    else
        J_fact = ldl_factorize!(Symmetric(J_augm, :U), J_fact)
    end
    J_fact.__factorized = true
    itd = iter_data(tmp_diag, # tmp diag
                    diag_Q, #diag_Q
                    J_augm, #J_augm
                    J_fact,
                    diagind_J, #diagind_J
                    zeros(T, id.n_low), # x_m_lvar
                    zeros(T, id.n_upp), # uvar_m_x
                    zeros(T, id.n_cols), # init Qx
                    zeros(T, id.n_cols), # init ATy
                    zeros(T, id.n_rows), # Ax
                    zero(T), #xTQx
                    zero(T), #cTx
                    zero(T), #pri_obj
                    zero(T), #dual_obj
                    zero(T), #μ
                    zero(T),#pdd
                    zeros(T, 6), #l_pdd
                    one(T) #mean_pdd
                    )

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
    sc = stop_crit(itd.pdd < ϵ.pdd && res.rbNorm < ϵ.tol_rb && res.rcNorm < ϵ.tol_rc, # optimal
                   false, # small_Δx
                   itd.μ < ϵ.μ, # small_μ
                   false # tired
                   )

    return regu, itd, ϵ, pad, pt, res, sc
end

function convert_types!(T :: DataType, pt :: point{T_old}, itd :: iter_data{T_old}, res :: residuals{T_old},
                        regu :: regularization{T_old}, pad :: preallocated_data{T_old}, T0 :: DataType) where {T_old<:Real}

   pt = convert(point{T}, pt)
   res = convert(residuals{T}, res)
   itd = convert(iter_data{T}, itd)
   regu = convert(regularization{T}, regu)
   if T == Float64 && T0 == Float64
       regu.ρ_min, regu.δ_min = T(sqrt(eps())*1e-5), T(sqrt(eps())*1e0)
   else
       regu.ρ_min, regu.δ_min = T(sqrt(eps(T))*1e1), T(sqrt(eps(T))*1e1)
   end
   pad = convert(preallocated_data{T}, pad)

   regu.ρ /= 10
   regu.δ /= 10

   return pt, itd, res, regu, pad
end
