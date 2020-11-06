function get_QM_data(QM :: AbstractNLPModel)
    T = eltype(QM.meta.lvar)
    IntData = QM_IntData(Int[], Int[], Int[], Int[],Int[], Int[], Int[], QM.meta.ncon,
                         QM.meta.nvar, 0, 0)
    Oc = zeros(T, IntData.n_cols)
    IntData.ilow, IntData.iupp = [QM.meta.ilow; QM.meta.irng], [QM.meta.iupp; QM.meta.irng] # finite bounds index
    IntData.n_low, IntData.n_upp = length(IntData.ilow), length(IntData.iupp) # number of finite constraints
    IntData.irng = QM.meta.irng
    @assert QM.meta.lcon == QM.meta.ucon # equality constraint (Ax=b)
    A = jac(QM, Oc)
    A = dropzeros!(A)
    Q = hess(QM, Oc)  # lower triangular
    Q = dropzeros!(Q)
    FloatData_T = QM_FloatData(T[], T[], QM.meta.lcon, grad(QM, Oc), obj(QM, Oc), QM.meta.lvar, QM.meta.uvar)
    IntData.Arows, IntData.Acols, FloatData_T.Avals = findnz(A)
    IntData.Qrows, IntData.Qcols, FloatData_T.Qvals = findnz(Q)
    return FloatData_T, IntData, T
end

function convert_FloatData(T :: DataType, FloatData_T0 :: QM_FloatData{T0}) where {T0<:Real}
    FloatData_T = QM_FloatData(Array{T}(FloatData_T0.Qvals),Array{T}(FloatData_T0.Avals),
                               Array{T}(FloatData_T0.b), Array{T}(FloatData_T0.c), T(FloatData_T0.c0),
                               Array{T}(FloatData_T0.lvar), Array{T}(FloatData_T0.uvar))
    return FloatData_T
end

function init_params(FloatData_T0 :: QM_FloatData{T0}, IntData :: QM_IntData,
                     ϵ_T :: tolerances{T}, ϵ :: tolerances{T0}) where{T<:Real, T0<:Real}

    FloatData_T = convert_FloatData(T, FloatData_T0)
    res = residuals(zeros(T, IntData.n_rows), zeros(T, IntData.n_cols), zero(T), zero(T), zero(T))
    # init regularization values
    regu = regularization(T(sqrt(eps())*1e5), T(sqrt(eps())*1e5), T(sqrt(eps(T))*1e0), T(sqrt(eps(T))*1e0))
    pad = preallocated_data(zeros(T, IntData.n_cols+IntData.n_rows+IntData.n_low+IntData.n_upp), # Δ_aff
                            zeros(T, IntData.n_cols+IntData.n_rows+IntData.n_low+IntData.n_upp), # Δ_cc
                            zeros(T, IntData.n_cols+IntData.n_rows+IntData.n_low+IntData.n_upp), # Δ
                            zeros(T, IntData.n_cols+IntData.n_rows), # Δ_xλ
                            zeros(T, IntData.n_low), # x_m_l_αΔ_aff
                            zeros(T, IntData.n_upp), # u_m_x_αΔ_aff
                            zeros(T, IntData.n_low), # s_l_αΔ_aff
                            zeros(T, IntData.n_upp), # s_u_αΔ_aff
                            zeros(T, IntData.n_low), # rxs_l
                            zeros(T, IntData.n_upp) #rxs_u
                            )
    tmp_diag = -T(1.0e-2) .* ones(T, IntData.n_cols)
    J_augmrows = vcat(IntData.Qcols, IntData.Acols, IntData.n_cols+1:IntData.n_cols+IntData.n_rows, 1:IntData.n_cols)
    J_augmcols = vcat(IntData.Qrows, IntData.Arows.+IntData.n_cols, IntData.n_cols+1:IntData.n_cols+IntData.n_rows,
                      1:IntData.n_cols)
    J_augmvals = vcat(.-FloatData_T.Qvals, FloatData_T.Avals, regu.δ.*ones(T, IntData.n_rows), tmp_diag)
    J_augm = sparse(J_augmrows, J_augmcols, J_augmvals)
    diagind_J = get_diag_sparseCSC(J_augm)
    J_fact = ldl_analyze(Symmetric(J_augm, :U))
    J_fact = ldl_factorize!(Symmetric(J_augm, :U), J_fact)
    J_fact.__factorized = true
    itd = iter_data(tmp_diag, # tmp diag
                    get_diag_sparseCOO(IntData.Qrows, IntData.Qcols, FloatData_T.Qvals, IntData.n_cols), #diag_Q
                    J_augm, #J_augm
                    J_fact, #J_fact
                    diagind_J, #diagind_J
                    zeros(T, IntData.n_low), # x_m_lvar
                    zeros(T, IntData.n_upp), # uvar_m_x
                    zeros(T, IntData.n_cols), # init Qx
                    zeros(T, IntData.n_cols), # init ATλ
                    zeros(T, IntData.n_rows), # Ax
                    zero(T), #xTQx
                    zero(T), #cTx
                    zero(T), #pri_obj
                    zero(T), #dual_obj
                    zero(T), #μ
                    zero(T),#pdd
                    zeros(T, 6), #l_pdd
                    one(T) #mean_pdd
                    )

    pt, itd, pad.Δ_xλ = @views starting_points(FloatData_T, IntData, itd, pad.Δ_xλ)

    # stopping criterion
    #     rcNorm, rbNorm = norm(rc), norm(rb)
    #     optimal = pdd < ϵ_pdd && rbNorm < ϵ_rb && rcNorm < ϵ_rc
    res.rb .= itd.Ax .- FloatData_T.b
    res.rc .= itd.ATλ .-itd.Qx .+ pt.s_l .- pt.s_u .- FloatData_T.c
    res.rcNorm, res.rbNorm = norm(res.rc, Inf), norm(res.rb, Inf)
    ϵ_T.tol_rb, ϵ_T.tol_rc = ϵ_T.rb*(one(T) + res.rbNorm), ϵ_T.rc*(one(T) + res.rcNorm)
    ϵ.tol_rb, ϵ.tol_rc = ϵ.rb*(one(T0) + T0(res.rbNorm)), ϵ.rc*(one(T0) + T0(res.rcNorm))
    sc = stop_crit(itd.pdd < ϵ_T.pdd && res.rbNorm < ϵ_T.tol_rb && res.rcNorm < ϵ_T.tol_rc, # optimal
                   false, # small_Δx
                   itd.μ < ϵ_T.μ, # small_μ
                   false # tired
                   )

    return FloatData_T, ϵ_T, ϵ, regu, itd, pad, pt, res, sc
end

function init_params_mono(FloatData_T :: QM_FloatData{T}, IntData :: QM_IntData, ϵ :: tolerances{T}) where {T<:Real}

    res = residuals(zeros(T, IntData.n_rows), zeros(T, IntData.n_cols), zero(T), zero(T), zero(T))
    # init regularization values
    regu = regularization(T(sqrt(eps())*1e5), T(sqrt(eps())*1e5), 1e-5*sqrt(eps(T)), 1e0*sqrt(eps(T)))
    pad = preallocated_data(zeros(T, IntData.n_cols+IntData.n_rows+IntData.n_low+IntData.n_upp), # Δ_aff
                            zeros(T, IntData.n_cols+IntData.n_rows+IntData.n_low+IntData.n_upp), # Δ_cc
                            zeros(T, IntData.n_cols+IntData.n_rows+IntData.n_low+IntData.n_upp), # Δ
                            zeros(T, IntData.n_cols+IntData.n_rows), # Δ_xλ
                            zeros(T, IntData.n_low), # x_m_l_αΔ_aff
                            zeros(T, IntData.n_upp), # u_m_x_αΔ_aff
                            zeros(T, IntData.n_low), # s_l_αΔ_aff
                            zeros(T, IntData.n_upp), # s_u_αΔ_aff
                            zeros(T, IntData.n_low), # rxs_l
                            zeros(T, IntData.n_upp) #rxs_u
                            )

    tmp_diag = -T(1.0e0)/2 .* ones(T, IntData.n_cols)
    J_augmrows = vcat(IntData.Qcols, IntData.Acols, IntData.n_cols+1:IntData.n_cols+IntData.n_rows,
                      1:IntData.n_cols)
    J_augmcols = vcat(IntData.Qrows, IntData.Arows.+IntData.n_cols, IntData.n_cols+1:IntData.n_cols+IntData.n_rows,
                      1:IntData.n_cols)
    J_augmvals = vcat(.-FloatData_T.Qvals, FloatData_T.Avals, regu.δ*ones(T, IntData.n_rows), tmp_diag)
    J_augm = sparse(J_augmrows, J_augmcols, J_augmvals)
    diagind_J = get_diag_sparseCSC(J_augm)
    J_fact = ldl_analyze(Symmetric(J_augm, :U))
    J_fact = ldl_factorize!(Symmetric(J_augm, :U), J_fact)
    J_fact.__factorized = true
    itd = iter_data(tmp_diag, # tmp diag
                    get_diag_sparseCOO(IntData.Qrows, IntData.Qcols, FloatData_T.Qvals, IntData.n_cols), #diag_Q
                    J_augm, #J_augm
                    J_fact,
                    diagind_J, #diagind_J
                    zeros(T, IntData.n_low), # x_m_lvar
                    zeros(T, IntData.n_upp), # uvar_m_x
                    zeros(T, IntData.n_cols), # init Qx
                    zeros(T, IntData.n_cols), # init ATλ
                    zeros(T, IntData.n_rows), # Ax
                    zero(T), #xTQx
                    zero(T), #cTx
                    zero(T), #pri_obj
                    zero(T), #dual_obj
                    zero(T), #μ
                    zero(T),#pdd
                    zeros(T, 6), #l_pdd
                    one(T) #mean_pdd
                    )

    pt, itd, pad.Δ_xλ = @views starting_points(FloatData_T, IntData, itd, pad.Δ_xλ)

    # stopping criterion
    #     rcNorm, rbNorm = norm(rc), norm(rb)
    #     optimal = pdd < ϵ_pdd && rbNorm < ϵ_rb && rcNorm < ϵ_rc
    res.rb .= itd.Ax .- FloatData_T.b
    res.rc .= itd.ATλ .-itd.Qx .+ pt.s_l .- pt.s_u .- FloatData_T.c
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
    itd.x_m_lvar, itd.uvar_m_x = convert(Array{T}, itd.x_m_lvar), convert(Array{T}, itd.uvar_m_x)
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
