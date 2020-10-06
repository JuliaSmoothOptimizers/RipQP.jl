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
    FloatData_T0 = QM_FloatData(T[], T[], QM.meta.lcon, grad(QM, Oc), obj(QM, Oc), QM.meta.lvar, QM.meta.uvar)
    IntData.Arows, IntData.Acols, FloatData_T0.Avals = findnz(A)
    IntData.Qrows, IntData.Qcols, FloatData_T0.Qvals = findnz(Q)
    return FloatData_T0, IntData, T
end

function init_params(T, FloatData_T0, IntData, ϵ_μ, ϵ_Δx)

    FloatData_T = QM_FloatData(Array{T}(FloatData_T0.Qvals),Array{T}(FloatData_T0.Avals),
                               Array{T}(FloatData_T0.b), Array{T}(FloatData_T0.c), T(FloatData_T0.c0),
                               Array{T}(FloatData_T0.lvar), Array{T}(FloatData_T0.uvar))
    res = residuals(T[], T[], T, T)
    ϵ_T = tolerances(T(1e-3), T(1e-4), T(1e-4), T(ϵ_μ), T(ϵ_Δx))
    # init regularization values
    regu = regularization(T(sqrt(eps())*1e5), T(sqrt(eps())*1e5), T(sqrt(eps(T))*1e0), T(sqrt(eps(T))*1e0))
    tmp_diag = -T(1.0e-2) .* ones(T, IntData.n_cols)
    J_augmrows = vcat(IntData.Qcols, IntData.Acols, IntData.n_cols+1:IntData.n_cols+IntData.n_rows,
                      1:IntData.n_cols)
    J_augmcols = vcat(IntData.Qrows, IntData.Arows.+IntData.n_cols, IntData.n_cols+1:IntData.n_cols+IntData.n_rows,
                      1:IntData.n_cols)
    J_augmvals = vcat(-FloatData_T.Qvals, FloatData_T.Avals, regu.δ.*ones(T, IntData.n_rows), tmp_diag)
    J_augm = sparse(J_augmrows, J_augmcols, J_augmvals)
    diagind_J = get_diag_sparseCSC(J_augm)
    diag_Q = get_diag_sparseCOO(IntData.Qrows, IntData.Qcols, FloatData_T.Qvals, IntData.n_cols)
    x_m_l_αΔ_aff = zeros(T, IntData.n_low) # x-lvar + αΔ_aff
    u_m_x_αΔ_aff = zeros(T, IntData.n_upp) # uvar-x + αΔ_aff
    s_l_αΔ_aff = zeros(T, IntData.n_low) # s_l + αΔ_aff
    s_u_αΔ_aff = zeros(T, IntData.n_upp) # s_l + αΔ_aff
    rxs_l, rxs_u = zeros(T, IntData.n_low), zeros(T, IntData.n_upp)
    Δ_aff = zeros(T, IntData.n_cols+IntData.n_rows+IntData.n_low+IntData.n_upp)
    Δ_cc = zeros(T, IntData.n_cols+IntData.n_rows+IntData.n_low+IntData.n_upp)
    Δ = zeros(T, IntData.n_cols+IntData.n_rows+IntData.n_low+IntData.n_upp)
    Δ_xλ = zeros(T, IntData.n_cols+IntData.n_rows)

    pt, J_fact, J_P, Qx, ATλ, x_m_lvar, uvar_m_x, Δ_xλ = @views starting_points(FloatData_T, IntData, J_augm , Δ_xλ)
    Qx = mul_Qx_COO!(Qx, IntData.Qrows, IntData.Qcols, FloatData_T.Qvals, pt.x)
    ATλ = mul_ATλ_COO!(ATλ, IntData.Arows, IntData.Acols, FloatData_T.Avals, pt.λ)
    Ax = zeros(T, IntData.n_rows)
    Ax = mul_Ax_COO!(Ax, IntData.Arows, IntData.Acols, FloatData_T.Avals, pt.x)
    res.rb = Ax - FloatData_T.b
    res.rc = -Qx + ATλ + pt.s_l - pt.s_u - FloatData_T.c
    x_m_lvar .= @views pt.x[IntData.ilow] .- FloatData_T.lvar[IntData.ilow]
    uvar_m_x .= @views FloatData_T.uvar[IntData.iupp] .- pt.x[IntData.iupp]

    # stopping criterion
    xTQx_2 = pt.x' * Qx / 2
    cTx = FloatData_T.c' * pt.x
    pri_obj = xTQx_2 + cTx + FloatData_T.c0
    dual_obj = FloatData_T.b' * pt.λ - xTQx_2 + view(pt.s_l, IntData.ilow)'*view(FloatData_T.lvar, IntData.ilow) -
                    view(pt.s_u, IntData.iupp)'*view(FloatData_T.uvar, IntData.iupp) + FloatData_T.c0
    μ = @views compute_μ(x_m_lvar, uvar_m_x, pt.s_l[IntData.ilow], pt.s_u[IntData.iupp], IntData.n_low, IntData.n_upp)
    pdd = abs(pri_obj - dual_obj ) / (one(T) + abs(pri_obj))
    #     rcNorm, rbNorm = norm(rc), norm(rb)
    #     optimal = pdd < ϵ_pdd && rbNorm < ϵ_rb && rcNorm < ϵ_rc
    res.rcNorm, res.rbNorm = norm(res.rc, Inf), norm(res.rb, Inf)
    tol_rb_T, tol_rc_T = ϵ_T.rb*(one(T) + res.rbNorm), ϵ_T.rc*(one(T) + res.rcNorm)
    tol_rb, tol_rc = ϵ_T.rb*(one(Float64) + Float64(res.rbNorm)), ϵ_T.rc*(one(Float64) + Float64(res.rcNorm))
    optimal = pdd < ϵ_T.pdd && res.rbNorm < tol_rb_T && res.rcNorm < tol_rc_T
    small_Δx, small_μ = false, μ < ϵ_T.μ
    l_pdd = zeros(T, 6)
    mean_pdd = one(T)
    n_Δx = zero(T)

    return FloatData_T, ϵ_T, regu, tmp_diag, J_augm, diagind_J, diag_Q, x_m_l_αΔ_aff, u_m_x_αΔ_aff,
                s_l_αΔ_aff, s_u_αΔ_aff, rxs_l, rxs_u, Δ_aff, Δ_cc, Δ, Δ_xλ, pt,
                J_fact, J_P, Qx, ATλ, Ax, x_m_lvar, uvar_m_x, xTQx_2,  cTx, pri_obj, dual_obj,
                μ, pdd, res, tol_rb_T, tol_rc_T,tol_rb, tol_rc,
                optimal, small_Δx, small_μ, l_pdd, mean_pdd, n_Δx
end

function init_params_mono(FloatData_T0, IntData, ϵ)
    T = eltype(FloatData_T0.Avals)
    res = residuals(T[], T[], T, T)
    # init regularization values
    regu = regularization(T(sqrt(eps())*1e5), T(sqrt(eps())*1e5), 1e-5*sqrt(eps(T)), 1e0*sqrt(eps(T)))
    tmp_diag = -T(1.0e0)/2 .* ones(T, IntData.n_cols)
    J_augmrows = vcat(IntData.Qcols, IntData.Acols, IntData.n_cols+1:IntData.n_cols+IntData.n_rows,
                      1:IntData.n_cols)
    J_augmcols = vcat(IntData.Qrows, IntData.Arows.+IntData.n_cols, IntData.n_cols+1:IntData.n_cols+IntData.n_rows,
                      1:IntData.n_cols)
    J_augmvals = vcat(-FloatData_T0.Qvals, FloatData_T0.Avals, regu.δ.*ones(T, IntData.n_rows), tmp_diag)
    J_augm = sparse(J_augmrows, J_augmcols, J_augmvals)
    diagind_J = get_diag_sparseCSC(J_augm)
    diag_Q = get_diag_sparseCOO(IntData.Qrows, IntData.Qcols, FloatData_T0.Qvals, IntData.n_cols)
    x_m_l_αΔ_aff = zeros(T, IntData.n_low) # x-lvar + αΔ_aff
    u_m_x_αΔ_aff = zeros(T, IntData.n_upp) # uvar-x + αΔ_aff
    s_l_αΔ_aff = zeros(T, IntData.n_low) # s_l + αΔ_aff
    s_u_αΔ_aff = zeros(T, IntData.n_upp) # s_l + αΔ_aff
    rxs_l, rxs_u = zeros(T, IntData.n_low), zeros(T, IntData.n_upp)
    Δ_aff = zeros(T, IntData.n_cols+IntData.n_rows+IntData.n_low+IntData.n_upp)
    Δ_cc = zeros(T, IntData.n_cols+IntData.n_rows+IntData.n_low+IntData.n_upp)
    Δ = zeros(T, IntData.n_cols+IntData.n_rows+IntData.n_low+IntData.n_upp)
    Δ_xλ = zeros(T, IntData.n_cols+IntData.n_rows)

    pt, J_fact, J_P, Qx, ATλ, x_m_lvar, uvar_m_x, Δ_xλ = @views starting_points(FloatData_T0, IntData, J_augm, Δ_xλ)
    Qx = mul_Qx_COO!(Qx, IntData.Qrows, IntData.Qcols, FloatData_T0.Qvals, pt.x)
    ATλ = mul_ATλ_COO!(ATλ, IntData.Arows, IntData.Acols, FloatData_T0.Avals, pt.λ)
    Ax = zeros(T, IntData.n_rows)
    Ax = mul_Ax_COO!(Ax, IntData.Arows, IntData.Acols, FloatData_T0.Avals, pt.x)
    res.rb = Ax - FloatData_T0.b
    res.rc = -Qx + ATλ + pt.s_l - pt.s_u - FloatData_T0.c
    x_m_lvar .= @views pt.x[IntData.ilow] .- FloatData_T0.lvar[IntData.ilow]
    uvar_m_x .= @views FloatData_T0.uvar[IntData.iupp] .- pt.x[IntData.iupp]

    # stopping criterion
    xTQx_2 = pt.x' * Qx / 2
    cTx = FloatData_T0.c' * pt.x
    pri_obj = xTQx_2 + cTx + FloatData_T0.c0
    dual_obj = FloatData_T0.b' * pt.λ - xTQx_2 + view(pt.s_l, IntData.ilow)'*view(FloatData_T0.lvar, IntData.ilow) -
                    view(pt.s_u, IntData.iupp)'*view(FloatData_T0.uvar, IntData.iupp) + FloatData_T0.c0
    μ = @views compute_μ(x_m_lvar, uvar_m_x, pt.s_l[IntData.ilow], pt.s_u[IntData.iupp], IntData.n_low, IntData.n_upp)
    pdd = abs(pri_obj - dual_obj ) / (one(T) + abs(pri_obj))
    #     rcNorm, rbNorm = norm(rc), norm(rb)
    #     optimal = pdd < ϵ_pdd && rbNorm < ϵ_rb && rcNorm < ϵ_rc
    res.rcNorm, res.rbNorm = norm(res.rc, Inf), norm(res.rb, Inf)
    tol_rb, tol_rc = ϵ.rb*(one(T) + res.rbNorm), ϵ.rc*(one(T) + res.rcNorm)
    optimal = pdd < ϵ.pdd && res.rbNorm < tol_rb && res.rcNorm < tol_rc
    small_Δx, small_μ = false, μ < ϵ.μ
    l_pdd = zeros(T, 6)
    mean_pdd = one(T)
    n_Δx = zero(T)
    return regu, tmp_diag, J_augm, diagind_J, diag_Q,
                x_m_l_αΔ_aff, u_m_x_αΔ_aff, s_l_αΔ_aff, s_u_αΔ_aff, rxs_l, rxs_u, Δ_aff,
                Δ_cc, Δ, Δ_xλ, pt, J_fact, J_P, Qx, ATλ, Ax, x_m_lvar, uvar_m_x,
                xTQx_2,  cTx, pri_obj, dual_obj, μ, pdd, res,
                tol_rb, tol_rc, optimal, small_Δx, small_μ, l_pdd, mean_pdd, n_Δx
end

function convert_types!(T, pt, x_m_lvar, uvar_m_x, res, Qx, ATλ, Ax, xTQx_2, cTx, pri_obj,
                        dual_obj, pdd, l_pdd, mean_pdd, n_Δx, μ, regu, J_augm, J_P,
                        J_fact, Δ_aff, Δ_cc, Δ, Δ_xλ, rxs_l, rxs_u, s_l_αΔ_aff,
                        s_u_αΔ_aff, x_m_l_αΔ_aff, u_m_x_αΔ_aff, diag_Q, tmp_diag)

   pt.x, pt.λ, pt.s_l, pt.s_u = convert(Array{T}, pt.x), convert(Array{T}, pt.λ),
                                    convert(Array{T}, pt.s_l), convert(Array{T}, pt.s_u)
   x_m_lvar, uvar_m_x = convert(Array{T}, x_m_lvar), convert(Array{T}, uvar_m_x)
   res.rc, res.rb = convert(Array{T}, res.rc), convert(Array{T}, res.rb)
   res.rcNorm, res.rbNorm = convert(T, res.rcNorm), convert(T, res.rbNorm)
   Qx, ATλ, Ax = convert(Array{T}, Qx), convert(Array{T}, ATλ), convert(Array{T}, Ax)
   xTQx_2, cTx = convert(T, xTQx_2), convert(T, cTx)
   pri_obj, dual_obj = convert(T, pri_obj), convert(T, dual_obj)
   pdd, l_pdd, mean_pdd = convert(T, pdd), convert(Array{T}, l_pdd), convert(T, mean_pdd)
   n_Δx, μ = convert(T, n_Δx), convert(T, μ)
   regu.ρ, regu.δ = convert(T, regu.ρ), convert(T, regu.δ)
   regu.ρ_min, regu.δ_min = T(sqrt(eps())*1e-5), T(sqrt(eps())*1e0)
   J_augm = convert(SparseMatrixCSC{T,Int64}, J_augm)
   J_P = LDLFactorizations.LDLFactorization(J_P.__analyzed, J_P.__factorized, J_P.__upper,
                                            J_P.n, J_P.parent, J_P.Lnz, J_P.flag, J_P.P,
                                            J_P.pinv, J_P.Lp, J_P.Cp, J_P.Ci, J_P.Li,
                                            Array{T}(J_P.Lx), Array{T}(J_P.d), Array{T}(J_P.Y), J_P.pattern)
   J_fact = LDLFactorizations.LDLFactorization(J_fact.__analyzed, J_fact.__factorized, J_fact.__upper,
                                               J_fact.n, J_fact.parent, J_fact.Lnz, J_fact.flag, J_fact.P,
                                               J_fact.pinv, J_fact.Lp, J_fact.Cp, J_fact.Ci, J_fact.Li,
                                               Array{T}(J_fact.Lx), Array{T}(J_fact.d), Array{T}(J_fact.Y), J_fact.pattern)
   Δ_aff, Δ_cc, Δ = convert(Array{T}, Δ_aff), convert(Array{T}, Δ_cc), convert(Array{T}, Δ)
   Δ_xλ, rxs_l, rxs_u = convert(Array{T}, Δ_xλ), convert(Array{T}, rxs_l), convert(Array{T}, rxs_u)
   s_l_αΔ_aff, s_u_αΔ_aff = convert(Array{T}, s_l_αΔ_aff), convert(Array{T}, s_u_αΔ_aff)
   x_m_l_αΔ_aff, u_m_x_αΔ_aff = convert(Array{T}, x_m_l_αΔ_aff), convert(Array{T}, u_m_x_αΔ_aff)
   diag_Q, tmp_diag = convert(Array{T}, diag_Q), convert(Array{T}, tmp_diag)

   return  pt, x_m_lvar, uvar_m_x, res, Qx, ATλ, Ax, xTQx_2, cTx, pri_obj,
                dual_obj, pdd, l_pdd, mean_pdd, n_Δx, μ, regu, J_augm, J_P,
                J_fact, Δ_aff, Δ_cc, Δ, Δ_xλ, rxs_l, rxs_u, s_l_αΔ_aff,
                s_u_αΔ_aff, x_m_l_αΔ_aff, u_m_x_αΔ_aff, diag_Q, tmp_diag
end
