function init_params(T, Qrows, Qcols, Qvals,  Arows, Acols, Avals, c, c0, b,
                     lvar, uvar, tol_Δx, ϵ_μ, ϵ_rb, ϵ_rc, n_rows, n_cols,
                     ilow, iupp, irng, n_low, n_upp)

    Qvals_T = Array{T}(Qvals)
    Avals_T = Array{T}(Avals)
    c_T = Array{T}(c)
    c0_T = T(c0)
    b_T = Array{T}(b)
    lvar_T = Array{T}(lvar)
    uvar_T = Array{T}(uvar)
    ϵ_pdd_T = T(1e-2)
    ϵ_rb_T = T(1e-2)
    ϵ_rc_T = T(1e-2)
    tol_Δx_T = T(tol_Δx)
    ϵ_μ_T = T(ϵ_μ)
    # init regularization values
    ρ, δ = T(sqrt(eps())*1e6), T(sqrt(eps())*1e8)
    ρ_min, δ_min = T(sqrt(eps(T))*1e0), T(sqrt(eps(T))*1e0)
    tmp_diag = -T(1.0e-2) .* ones(T, n_cols)
    J_augmvals = vcat(-Qvals_T, Avals_T, δ.*ones(T, n_rows), tmp_diag)
    J_augmrows = vcat(Qcols, Acols, n_cols+1:n_cols+n_rows, 1:n_cols)
    J_augmcols = vcat(Qrows, Arows.+n_cols, n_cols+1:n_cols+n_rows, 1:n_cols)
    J_augmvals = vcat(-Qvals_T, Avals_T, δ.*ones(T, n_rows), tmp_diag)
    J_augm = sparse(J_augmrows, J_augmcols, J_augmvals)
    diagind_J = get_diag_sparseCSC(J_augm)
    diag_Q = get_diag_sparseCOO(Qrows, Qcols, Qvals_T, n_cols)
    J_augmrows = vcat(Qcols, Acols, n_cols+1:n_cols+n_rows, 1:n_cols)
    J_augmcols = vcat(Qrows, Arows.+n_cols, n_cols+1:n_cols+n_rows, 1:n_cols)
    J_augmvals = vcat(-Qvals_T, Avals_T, δ.*ones(T, n_rows), tmp_diag)
    J_augm = sparse(J_augmrows, J_augmcols, J_augmvals)
    diagind_J = get_diag_sparseCSC(J_augm)
    diag_Q = get_diag_sparseCOO(Qrows, Qcols, Qvals_T, n_cols)
    x_m_l_αΔ_aff = zeros(T, n_low) # x-lvar + αΔ_aff
    u_m_x_αΔ_aff = zeros(T, n_upp) # uvar-x + αΔ_aff
    s_l_αΔ_aff = zeros(T, n_low) # s_l + αΔ_aff
    s_u_αΔ_aff = zeros(T, n_upp) # s_l + αΔ_aff
    rxs_l, rxs_u = zeros(T, n_low), zeros(T, n_upp)
    Δ_aff = zeros(T, n_cols+n_rows+n_low+n_upp)
    Δ_cc = zeros(T, n_cols+n_rows+n_low+n_upp)
    Δ = zeros(T, n_cols+n_rows+n_low+n_upp)
    Δ_xλ = zeros(T, n_cols+n_rows)

    x, λ, s_l, s_u, J_fact, J_P, Qx, ATλ,
    x_m_lvar, uvar_m_x, Δ_xλ = @views starting_points(Qrows, Qcols, Qvals_T, Arows, Acols, Avals_T,
                                                      b_T, c_T, lvar_T, uvar_T, ilow, iupp, irng,
                                                      J_augm , n_rows, n_cols, Δ_xλ)
    Qx = mul_Qx_COO!(Qx, Qrows, Qcols, Qvals_T, x)
    ATλ = mul_ATλ_COO!(ATλ, Arows, Acols, Avals_T, λ)
    Ax = zeros(T,  n_rows)
    Ax = mul_Ax_COO!(Ax, Arows, Acols, Avals_T, x)
    rb = Ax - b_T
    rc = -Qx + ATλ + s_l - s_u - c_T
    x_m_lvar .= @views x[ilow] .- lvar_T[ilow]
    uvar_m_x .= @views uvar_T[iupp] .- x[iupp]

    # stopping criterion
    xTQx_2 = x' * Qx / 2
    cTx = c_T' * x
    pri_obj = xTQx_2 + cTx + c0_T
    dual_obj = b_T' * λ - xTQx_2 + view(s_l,ilow)'*view(lvar_T,ilow) -
    view(s_u,iupp)'*view(uvar_T,iupp) +c0_T
    μ = @views compute_μ(x_m_lvar, uvar_m_x, s_l[ilow], s_u[iupp], n_low, n_upp)
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

    return Qvals_T, Avals_T, c_T, c0_T, b_T, lvar_T, uvar_T, ϵ_pdd_T, ϵ_rb_T, ϵ_rc_T,
                tol_Δx_T, ϵ_μ_T, ρ, δ, ρ_min, δ_min, tmp_diag, J_augm, diagind_J, diag_Q,
                x_m_l_αΔ_aff, u_m_x_αΔ_aff, s_l_αΔ_aff, s_u_αΔ_aff, rxs_l, rxs_u, Δ_aff,
                Δ_cc, Δ, Δ_xλ, x, λ, s_l, s_u, J_fact, J_P, Qx, ATλ, Ax, x_m_lvar, uvar_m_x,
                xTQx_2,  cTx, pri_obj, dual_obj, μ, pdd, rc, rb, rcNorm, rbNorm, tol_rb_T, tol_rc_T,
                tol_rb, tol_rc, optimal, small_Δx, small_μ, l_pdd, mean_pdd, n_Δx
end

function init_params_mono(Qrows, Qcols, Qvals,  Arows, Acols, Avals, c, c0, b,
                          lvar, uvar, tol_Δx, ϵ_pdd, ϵ_μ, ϵ_rb, ϵ_rc, n_rows, n_cols,
                          ilow, iupp, irng, n_low, n_upp)
    T = eltype(Avals)
    # init regularization values
    ρ, δ = T(sqrt(eps())*1e5), T(sqrt(eps())*1e5)
    ρ_min, δ_min =  1e-5*sqrt(eps(T)), 1e0*sqrt(eps(T))
    tmp_diag = -T(1.0e0)/2 .* ones(T, n_cols)
    J_augmvals = vcat(-Qvals, Avals, δ.*ones(T, n_rows), tmp_diag)
    J_augmrows = vcat(Qcols, Acols, n_cols+1:n_cols+n_rows, 1:n_cols)
    J_augmcols = vcat(Qrows, Arows.+n_cols, n_cols+1:n_cols+n_rows, 1:n_cols)
    J_augmvals = vcat(-Qvals, Avals, δ.*ones(T, n_rows), tmp_diag)
    J_augm = sparse(J_augmrows, J_augmcols, J_augmvals)
    diagind_J = get_diag_sparseCSC(J_augm)
    diag_Q = get_diag_sparseCOO(Qrows, Qcols, Qvals, n_cols)
    J_augmrows = vcat(Qcols, Acols, n_cols+1:n_cols+n_rows, 1:n_cols)
    J_augmcols = vcat(Qrows, Arows.+n_cols, n_cols+1:n_cols+n_rows, 1:n_cols)
    J_augmvals = vcat(-Qvals, Avals, δ.*ones(T, n_rows), tmp_diag)
    J_augm = sparse(J_augmrows, J_augmcols, J_augmvals)
    diagind_J = get_diag_sparseCSC(J_augm)
    diag_Q = get_diag_sparseCOO(Qrows, Qcols, Qvals, n_cols)
    x_m_l_αΔ_aff = zeros(T, n_low) # x-lvar + αΔ_aff
    u_m_x_αΔ_aff = zeros(T, n_upp) # uvar-x + αΔ_aff
    s_l_αΔ_aff = zeros(T, n_low) # s_l + αΔ_aff
    s_u_αΔ_aff = zeros(T, n_upp) # s_l + αΔ_aff
    rxs_l, rxs_u = zeros(T, n_low), zeros(T, n_upp)
    Δ_aff = zeros(T, n_cols+n_rows+n_low+n_upp)
    Δ_cc = zeros(T, n_cols+n_rows+n_low+n_upp)
    Δ = zeros(T, n_cols+n_rows+n_low+n_upp)
    Δ_xλ = zeros(T, n_cols+n_rows)

    x, λ, s_l, s_u, J_fact, J_P, Qx, ATλ,
        x_m_lvar, uvar_m_x, Δ_xλ = @views starting_points(Qrows, Qcols, Qvals, Arows, Acols, Avals,
                                                          b, c, lvar, uvar, ilow, iupp, irng,
                                                          J_augm , n_rows, n_cols, Δ_xλ)
    Qx = mul_Qx_COO!(Qx, Qrows, Qcols, Qvals, x)
    ATλ = mul_ATλ_COO!(ATλ, Arows, Acols, Avals, λ)
    Ax = zeros(T, n_rows)
    Ax = mul_Ax_COO!(Ax, Arows, Acols, Avals, x)
    rb = Ax - b
    rc = -Qx + ATλ + s_l - s_u - c
    x_m_lvar .= @views x[ilow] .- lvar[ilow]
    uvar_m_x .= @views uvar[iupp] .- x[iupp]

    # stopping criterion
    xTQx_2 = x' * Qx / 2
    cTx = c' * x
    pri_obj = xTQx_2 + cTx + c0
    dual_obj = b' * λ - xTQx_2 + view(s_l, ilow)'*view(lvar, ilow) -
                    view(s_u, iupp)'*view(uvar,iupp) +c0
    μ = @views compute_μ(x_m_lvar, uvar_m_x, s_l[ilow], s_u[iupp], n_low, n_upp)
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
