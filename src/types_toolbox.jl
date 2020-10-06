mutable struct QM_FloatData{T}
    Qvals :: Vector{T}
    Avals :: Vector{T}
    b     :: Vector{T}
    c     :: Vector{T}
    c0    :: T
    lvar  :: Vector{T}
    uvar  :: Vector{T}
end

mutable struct QM_IntData
    Qrows  :: Vector{Int}
    Qcols  :: Vector{Int}
    Arows  :: Vector{Int}
    Acols  :: Vector{Int}
    ilow   :: Vector{Int}
    iupp   :: Vector{Int}
    irng   :: Vector{Int}
    n_rows :: Int
    n_cols :: Int
    n_low  :: Int
    n_upp  :: Int
end

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


function init_params(T, FloatData_T0, IntData, tol_Δx, ϵ_μ, ϵ_rb, ϵ_rc)

    FloatData_T = QM_FloatData(Array{T}(FloatData_T0.Qvals),Array{T}(FloatData_T0.Avals),
                               Array{T}(FloatData_T0.b), Array{T}(FloatData_T0.c), T(FloatData_T0.c0),
                               Array{T}(FloatData_T0.lvar), Array{T}(FloatData_T0.uvar))
    ϵ_pdd_T = T(1e-3)
    ϵ_rb_T = T(1e-4)
    ϵ_rc_T = T(1e-4)
    tol_Δx_T = T(tol_Δx)
    ϵ_μ_T = T(ϵ_μ)
    # init regularization values
    ρ, δ = T(sqrt(eps())*1e5), T(sqrt(eps())*1e5)
    ρ_min, δ_min = T(sqrt(eps(T))*1e0), T(sqrt(eps(T))*1e0)
    tmp_diag = -T(1.0e-2) .* ones(T, IntData.n_cols)
    J_augmrows = vcat(IntData.Qcols, IntData.Acols, IntData.n_cols+1:IntData.n_cols+IntData.n_rows,
                      1:IntData.n_cols)
    J_augmcols = vcat(IntData.Qrows, IntData.Arows.+IntData.n_cols, IntData.n_cols+1:IntData.n_cols+IntData.n_rows,
                      1:IntData.n_cols)
    J_augmvals = vcat(-FloatData_T.Qvals, FloatData_T.Avals, δ.*ones(T, IntData.n_rows), tmp_diag)
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

    x, λ, s_l, s_u, J_fact, J_P, Qx, ATλ,
    x_m_lvar, uvar_m_x, Δ_xλ = @views starting_points(FloatData_T, IntData, J_augm , Δ_xλ)
    Qx = mul_Qx_COO!(Qx, IntData.Qrows, IntData.Qcols, FloatData_T.Qvals, x)
    ATλ = mul_ATλ_COO!(ATλ, IntData.Arows, IntData.Acols, FloatData_T.Avals, λ)
    Ax = zeros(T, IntData.n_rows)
    Ax = mul_Ax_COO!(Ax, IntData.Arows, IntData.Acols, FloatData_T.Avals, x)
    rb = Ax - FloatData_T.b
    rc = -Qx + ATλ + s_l - s_u - FloatData_T.c
    x_m_lvar .= @views x[IntData.ilow] .- FloatData_T.lvar[IntData.ilow]
    uvar_m_x .= @views FloatData_T.uvar[IntData.iupp] .- x[IntData.iupp]

    # stopping criterion
    xTQx_2 = x' * Qx / 2
    cTx = FloatData_T.c' * x
    pri_obj = xTQx_2 + cTx + FloatData_T.c0
    dual_obj = FloatData_T.b' * λ - xTQx_2 + view(s_l, IntData.ilow)'*view(FloatData_T.lvar, IntData.ilow) -
                    view(s_u, IntData.iupp)'*view(FloatData_T.uvar, IntData.iupp) + FloatData_T.c0
    μ = @views compute_μ(x_m_lvar, uvar_m_x, s_l[IntData.ilow], s_u[IntData.iupp], IntData.n_low, IntData.n_upp)
    pdd = abs(pri_obj - dual_obj ) / (one(T) + abs(pri_obj))
    #     rcNorm, rbNorm = norm(rc), norm(rb)
    #     optimal = pdd < ϵ_pdd && rbNorm < ϵ_rb && rcNorm < ϵ_rc
    rcNorm, rbNorm = norm(rc, Inf), norm(rb, Inf)
    tol_rb_T, tol_rc_T = ϵ_rb_T*(one(T) + rbNorm), ϵ_rc_T*(one(T) + rcNorm)
    tol_rb, tol_rc = ϵ_rb*(one(Float64) + Float64(rbNorm)), ϵ_rc*(one(Float64) + Float64(rcNorm))
    optimal = pdd < ϵ_pdd_T && rbNorm < tol_rb_T && rcNorm < tol_rc_T
    small_Δx, small_μ = false, μ < ϵ_μ_T
    l_pdd = zeros(T, 6)
    mean_pdd = one(T)
    n_Δx = zero(T)

    return FloatData_T, ϵ_pdd_T, ϵ_rb_T, ϵ_rc_T, tol_Δx_T, ϵ_μ_T, ρ, δ, ρ_min,
                δ_min, tmp_diag, J_augm, diagind_J, diag_Q, x_m_l_αΔ_aff, u_m_x_αΔ_aff,
                s_l_αΔ_aff, s_u_αΔ_aff, rxs_l, rxs_u, Δ_aff, Δ_cc, Δ, Δ_xλ, x, λ, s_l, s_u,
                J_fact, J_P, Qx, ATλ, Ax, x_m_lvar, uvar_m_x, xTQx_2,  cTx, pri_obj, dual_obj,
                μ, pdd, rc, rb, rcNorm, rbNorm, tol_rb_T, tol_rc_T,tol_rb, tol_rc,
                optimal, small_Δx, small_μ, l_pdd, mean_pdd, n_Δx
end

function init_params_mono(FloatData_T0, IntData, tol_Δx, ϵ_pdd, ϵ_μ, ϵ_rb, ϵ_rc)
    T = eltype(FloatData_T0.Avals)
    # init regularization values
    ρ, δ = T(sqrt(eps())*1e5), T(sqrt(eps())*1e5)
    ρ_min, δ_min =  1e-5*sqrt(eps(T)), 1e0*sqrt(eps(T))
    tmp_diag = -T(1.0e0)/2 .* ones(T, IntData.n_cols)
    J_augmrows = vcat(IntData.Qcols, IntData.Acols, IntData.n_cols+1:IntData.n_cols+IntData.n_rows,
                      1:IntData.n_cols)
    J_augmcols = vcat(IntData.Qrows, IntData.Arows.+IntData.n_cols, IntData.n_cols+1:IntData.n_cols+IntData.n_rows,
                      1:IntData.n_cols)
    J_augmvals = vcat(-FloatData_T0.Qvals, FloatData_T0.Avals, δ.*ones(T, IntData.n_rows), tmp_diag)
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

    x, λ, s_l, s_u, J_fact, J_P, Qx, ATλ,
        x_m_lvar, uvar_m_x, Δ_xλ = @views starting_points(FloatData_T0, IntData, J_augm, Δ_xλ)
    Qx = mul_Qx_COO!(Qx, IntData.Qrows, IntData.Qcols, FloatData_T0.Qvals, x)
    ATλ = mul_ATλ_COO!(ATλ, IntData.Arows, IntData.Acols, FloatData_T0.Avals, λ)
    Ax = zeros(T, IntData.n_rows)
    Ax = mul_Ax_COO!(Ax, IntData.Arows, IntData.Acols, FloatData_T0.Avals, x)
    rb = Ax - FloatData_T0.b
    rc = -Qx + ATλ + s_l - s_u - FloatData_T0.c
    x_m_lvar .= @views x[IntData.ilow] .- FloatData_T0.lvar[IntData.ilow]
    uvar_m_x .= @views FloatData_T0.uvar[IntData.iupp] .- x[IntData.iupp]

    # stopping criterion
    xTQx_2 = x' * Qx / 2
    cTx = FloatData_T0.c' * x
    pri_obj = xTQx_2 + cTx + FloatData_T0.c0
    dual_obj = FloatData_T0.b' * λ - xTQx_2 + view(s_l, IntData.ilow)'*view(FloatData_T0.lvar, IntData.ilow) -
                    view(s_u, IntData.iupp)'*view(FloatData_T0.uvar, IntData.iupp) + FloatData_T0.c0
    μ = @views compute_μ(x_m_lvar, uvar_m_x, s_l[IntData.ilow], s_u[IntData.iupp], IntData.n_low, IntData.n_upp)
    pdd = abs(pri_obj - dual_obj ) / (one(T) + abs(pri_obj))
    #     rcNorm, rbNorm = norm(rc), norm(rb)
    #     optimal = pdd < ϵ_pdd && rbNorm < ϵ_rb && rcNorm < ϵ_rc
    rcNorm, rbNorm = norm(rc, Inf), norm(rb, Inf)
    tol_rb, tol_rc = ϵ_rb*(one(T) + rbNorm), ϵ_rc*(one(T) + rcNorm)
    optimal = pdd < ϵ_pdd && rbNorm < tol_rb && rcNorm < tol_rc
    small_Δx, small_μ = false, μ < ϵ_μ
    l_pdd = zeros(T, 6)
    mean_pdd = one(T)
    n_Δx = zero(T)
    return ρ, δ, ρ_min, δ_min, tmp_diag, J_augm, diagind_J, diag_Q,
                x_m_l_αΔ_aff, u_m_x_αΔ_aff, s_l_αΔ_aff, s_u_αΔ_aff, rxs_l, rxs_u, Δ_aff,
                Δ_cc, Δ, Δ_xλ, x, λ, s_l, s_u, J_fact, J_P, Qx, ATλ, Ax, x_m_lvar, uvar_m_x,
                xTQx_2,  cTx, pri_obj, dual_obj, μ, pdd, rc, rb, rcNorm, rbNorm,
                tol_rb, tol_rc, optimal, small_Δx, small_μ, l_pdd, mean_pdd, n_Δx
end

function convert_types!(T, x, λ, s_l, s_u, x_m_lvar, uvar_m_x, rc, rb,
                        rcNorm, rbNorm, Qx, ATλ, Ax, xTQx_2, cTx, pri_obj,
                        dual_obj, pdd, l_pdd, mean_pdd, n_Δx, μ, ρ, δ, J_augm, J_P,
                        J_fact, Δ_aff, Δ_cc, Δ, Δ_xλ, rxs_l, rxs_u, s_l_αΔ_aff,
                        s_u_αΔ_aff, x_m_l_αΔ_aff, u_m_x_αΔ_aff, diag_Q, tmp_diag,
                        ρ_min, δ_min)

   x, λ, s_l, s_u = convert(Array{T}, x), convert(Array{T}, λ), convert(Array{T}, s_l), convert(Array{T}, s_u)
   x_m_lvar, uvar_m_x = convert(Array{T}, x_m_lvar), convert(Array{T}, uvar_m_x)
   rc, rb = convert(Array{T}, rc), convert(Array{T}, rb)
   rcNorm, rbNorm = convert(T, rcNorm), convert(T, rbNorm)
   Qx, ATλ, Ax = convert(Array{T}, Qx), convert(Array{T}, ATλ), convert(Array{T}, Ax)
   xTQx_2, cTx = convert(T, xTQx_2), convert(T, cTx)
   pri_obj, dual_obj = convert(T, pri_obj), convert(T, dual_obj)
   pdd, l_pdd, mean_pdd = convert(T, pdd), convert(Array{T}, l_pdd), convert(T, mean_pdd)
   n_Δx, μ = convert(T, n_Δx), convert(T, μ)
   ρ, δ = convert(T, ρ), convert(T, δ)
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
   ρ_min, δ_min = T(sqrt(eps())*1e-5), T(sqrt(eps())*1e0)

   return  x, λ, s_l, s_u, x_m_lvar, uvar_m_x, rc, rb,
                rcNorm, rbNorm, Qx, ATλ, Ax, xTQx_2, cTx, pri_obj,
                dual_obj, pdd, l_pdd, mean_pdd, n_Δx, μ, ρ, δ, J_augm, J_P,
                J_fact, Δ_aff, Δ_cc, Δ, Δ_xλ, rxs_l, rxs_u, s_l_αΔ_aff,
                s_u_αΔ_aff, x_m_l_αΔ_aff, u_m_x_αΔ_aff, diag_Q, tmp_diag,
                ρ_min, δ_min
end
