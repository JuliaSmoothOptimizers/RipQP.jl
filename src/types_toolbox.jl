function get_QM_data(QM)
    T = eltype(QM.meta.lvar)
    IntData = QM_IntData(Int[], Int[], Int[], Int[],Int[], Int[], Int[], length(QM.meta.lcon),
                         length(QM.meta.lvar), 0, 0)
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

function convert_FloatData(T, FloatData_T0)
    FloatData_T = QM_FloatData(Array{T}(FloatData_T0.Qvals),Array{T}(FloatData_T0.Avals),
                               Array{T}(FloatData_T0.b), Array{T}(FloatData_T0.c), T(FloatData_T0.c0),
                               Array{T}(FloatData_T0.lvar), Array{T}(FloatData_T0.uvar))
    return FloatData_T
end

function init_params(T, T0, FloatData_T0, IntData, ϵ_T, ϵ)

    FloatData_T = convert_FloatData(T, FloatData_T0)
    res = residuals(T[], T[], zero(T), zero(T), zero(T))
    # init regularization values
    regu = regularization(T(sqrt(eps())*1e5), T(sqrt(eps())*1e5), T(sqrt(eps(T))*1e0), T(sqrt(eps(T))*1e0))
    itd = iter_data(-T(1.0e-2) .* ones(T, IntData.n_cols), # tmp diag
                    get_diag_sparseCOO(IntData.Qrows, IntData.Qcols, FloatData_T.Qvals, IntData.n_cols), #diag_Q
                    spzeros(IntData.n_cols+IntData.n_rows, IntData.n_cols+IntData.n_rows), #J_augm
                    spzeros(IntData.n_cols+IntData.n_rows, IntData.n_cols+IntData.n_rows), #J_fact
                    Int[],  #J_P
                    zeros(Int, IntData.n_cols+IntData.n_rows), #diagind_J
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
    J_augmrows = vcat(IntData.Qcols, IntData.Acols, IntData.n_cols+1:IntData.n_cols+IntData.n_rows,
                      1:IntData.n_cols)
    J_augmcols = vcat(IntData.Qrows, IntData.Arows.+IntData.n_cols, IntData.n_cols+1:IntData.n_cols+IntData.n_rows,
                      1:IntData.n_cols)
    J_augmvals = vcat(-FloatData_T.Qvals, FloatData_T.Avals, regu.δ.*ones(T, IntData.n_rows), itd.tmp_diag)
    itd.J_augm = sparse(J_augmrows, J_augmcols, J_augmvals)
    itd.diagind_J = get_diag_sparseCSC(itd.J_augm)

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

    pt, itd, pad.Δ_xλ = @views starting_points(FloatData_T, IntData, itd, pad.Δ_xλ)

    # stopping criterion
    #     rcNorm, rbNorm = norm(rc), norm(rb)
    #     optimal = pdd < ϵ_pdd && rbNorm < ϵ_rb && rcNorm < ϵ_rc
    res.rb = itd.Ax - FloatData_T.b
    res.rc = -itd.Qx + itd.ATλ + pt.s_l - pt.s_u - FloatData_T.c
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

function init_params_mono(FloatData_T, IntData, ϵ)

    T = eltype(FloatData_T.Avals)
    res = residuals(T[], T[], zero(T), zero(T), zero(T))
    # init regularization values
    regu = regularization(T(sqrt(eps())*1e5), T(sqrt(eps())*1e5), 1e-5*sqrt(eps(T)), 1e0*sqrt(eps(T)))
    itd = iter_data(-T(1.0e0)/2 .* ones(T, IntData.n_cols), # tmp diag
                    get_diag_sparseCOO(IntData.Qrows, IntData.Qcols, FloatData_T.Qvals, IntData.n_cols), #diag_Q
                    spzeros(IntData.n_cols+IntData.n_rows, IntData.n_cols+IntData.n_rows), #J_augm
                    spzeros(IntData.n_cols+IntData.n_rows, IntData.n_cols+IntData.n_rows), #J_fact
                    Int[],  #J_P
                    zeros(Int, IntData.n_cols+IntData.n_rows), #diagind_J
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
    J_augmrows = vcat(IntData.Qcols, IntData.Acols, IntData.n_cols+1:IntData.n_cols+IntData.n_rows,
                      1:IntData.n_cols)
    J_augmcols = vcat(IntData.Qrows, IntData.Arows.+IntData.n_cols, IntData.n_cols+1:IntData.n_cols+IntData.n_rows,
                      1:IntData.n_cols)
    J_augmvals = vcat(-FloatData_T.Qvals, FloatData_T.Avals, regu.δ.*ones(T, IntData.n_rows), itd.tmp_diag)
    itd.J_augm = sparse(J_augmrows, J_augmcols, J_augmvals)
    itd.diagind_J = get_diag_sparseCSC(itd.J_augm)

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

    pt, itd, pad.Δ_xλ = @views starting_points(FloatData_T, IntData, itd, pad.Δ_xλ)

    # stopping criterion
    #     rcNorm, rbNorm = norm(rc), norm(rb)
    #     optimal = pdd < ϵ_pdd && rbNorm < ϵ_rb && rcNorm < ϵ_rc
    res.rb = itd.Ax - FloatData_T.b
    res.rc = -itd.Qx + itd.ATλ + pt.s_l - pt.s_u - FloatData_T.c
    res.rcNorm, res.rbNorm = norm(res.rc, Inf), norm(res.rb, Inf)
    ϵ.tol_rb, ϵ.tol_rc = ϵ.rb*(one(T) + res.rbNorm), ϵ.rc*(one(T) + res.rcNorm)
    sc = stop_crit(itd.pdd < ϵ.pdd && res.rbNorm < ϵ.tol_rb && res.rcNorm < ϵ.tol_rc, # optimal
                   false, # small_Δx
                   itd.μ < ϵ.μ, # small_μ
                   false # tired
                   )

    return regu, itd, ϵ, pad, pt, res, sc
end

function convert_types!(T, pt, itd, res, regu, pad, T0)

   pt.x, pt.λ, pt.s_l, pt.s_u = convert(Array{T}, pt.x), convert(Array{T}, pt.λ),
                                    convert(Array{T}, pt.s_l), convert(Array{T}, pt.s_u)
   itd.x_m_lvar, itd.uvar_m_x = convert(Array{T}, itd.x_m_lvar), convert(Array{T}, itd.uvar_m_x)
   res.rc, res.rb = convert(Array{T}, res.rc), convert(Array{T}, res.rb)
   res.rcNorm, res.rbNorm = convert(T, res.rcNorm), convert(T, res.rbNorm)
   itd.Qx, itd.ATλ, itd.Ax = convert(Array{T}, itd.Qx), convert(Array{T}, itd.ATλ), convert(Array{T}, itd.Ax)
   itd.xTQx_2, itd.cTx = convert(T, itd.xTQx_2), convert(T, itd.cTx)
   itd.pri_obj, itd.dual_obj = convert(T, itd.pri_obj), convert(T, itd.dual_obj)
   itd.pdd, itd.l_pdd, itd.mean_pdd = convert(T, itd.pdd), convert(Array{T}, itd.l_pdd), convert(T, itd.mean_pdd)
   res.n_Δx, itd.μ = convert(T, res.n_Δx), convert(T, itd.μ)
   regu.ρ, regu.δ = convert(T, regu.ρ), convert(T, regu.δ)
   if T == Float64 && T0 == Float64
       regu.ρ_min, regu.δ_min = T(sqrt(eps())*1e-5), T(sqrt(eps())*1e0)
   else
       regu.ρ_min, regu.δ_min = T(sqrt(eps(T))*1e-3), T(sqrt(eps(T))*1e2)
   end
   itd.J_augm = convert(SparseMatrixCSC{T,Int64}, itd.J_augm)
   itd.J_P = LDLFactorizations.LDLFactorization(itd.J_P.__analyzed, itd.J_P.__factorized, itd.J_P.__upper,
                                                itd.J_P.n, itd.J_P.parent, itd.J_P.Lnz, itd.J_P.flag, itd.J_P.P,
                                                itd.J_P.pinv, itd.J_P.Lp, itd.J_P.Cp, itd.J_P.Ci, itd.J_P.Li,
                                                Array{T}(itd.J_P.Lx), Array{T}(itd.J_P.d), Array{T}(itd.J_P.Y),
                                                itd.J_P.pattern)
   itd.J_fact = LDLFactorizations.LDLFactorization(itd.J_fact.__analyzed, itd.J_fact.__factorized, itd.J_fact.__upper,
                                                   itd.J_fact.n, itd.J_fact.parent, itd.J_fact.Lnz, itd.J_fact.flag, itd.J_fact.P,
                                                   itd.J_fact.pinv, itd.J_fact.Lp, itd.J_fact.Cp, itd.J_fact.Ci, itd.J_fact.Li,
                                                   Array{T}(itd.J_fact.Lx), Array{T}(itd.J_fact.d), Array{T}(itd.J_fact.Y),
                                                   itd.J_fact.pattern)
   pad.Δ_aff, pad.Δ_cc, pad.Δ = convert(Array{T}, pad.Δ_aff), convert(Array{T}, pad.Δ_cc), convert(Array{T}, pad.Δ)
   pad.Δ_xλ, pad.rxs_l, pad.rxs_u = convert(Array{T}, pad.Δ_xλ), convert(Array{T}, pad.rxs_l), convert(Array{T}, pad.rxs_u)
   pad.s_l_αΔ_aff, pad.s_u_αΔ_aff = convert(Array{T}, pad.s_l_αΔ_aff), convert(Array{T}, pad.s_u_αΔ_aff)
   pad.x_m_l_αΔ_aff, pad.u_m_x_αΔ_aff = convert(Array{T}, pad.x_m_l_αΔ_aff), convert(Array{T}, pad.u_m_x_αΔ_aff)
   itd.diag_Q, itd.tmp_diag = convert(Array{T}, itd.diag_Q), convert(Array{T}, itd.tmp_diag)

   regu.ρ /= 10
   regu.δ /= 10

   return pt, itd, res, regu, pad
end
