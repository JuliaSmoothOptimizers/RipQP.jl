
function starting_points(FloatData, IntData, J_augm, Δ_xλ)

    T = eltype(FloatData.Avals)
    J_P = ldl_analyze(Symmetric(J_augm, :U))
    J_fact = ldl_factorize!(Symmetric(J_augm, :U), J_P)
    Δ_xλ[IntData.n_cols+1: end] = FloatData.b
    Δ_xλ = ldiv!(J_fact, Δ_xλ)
    pt0 = point(Δ_xλ[1:IntData.n_cols], Δ_xλ[IntData.n_cols+1:end], zeros(T, IntData.n_cols), zeros(T, IntData.n_cols))
    Qx, ATλ = zeros(T, IntData.n_cols), zeros(T, IntData.n_cols)
    Qx = mul_Qx_COO!(Qx, IntData.Qrows, IntData.Qcols, FloatData.Qvals, pt0.x)
    ATλ = mul_ATλ_COO!(ATλ, IntData.Arows, IntData.Acols, FloatData.Avals, pt0.λ)
    dual_val = Qx - ATλ + FloatData.c
    pt0.s_l[IntData.ilow] = @views dual_val[IntData.ilow]
    pt0.s_u[IntData.iupp] = @views -dual_val[IntData.iupp]
    x0_m_lvar = @views pt0.x[IntData.ilow] - FloatData.lvar[IntData.ilow]
    uvar_m_x0 = @views FloatData.uvar[IntData.iupp] - pt0.x[IntData.iupp]
    if IntData.n_low == 0
        δx_l1, δs_l1 = zero(T), zero(T)
    else
        δx_l1 = max(-T(1.5)*minimum(x0_m_lvar), T(1.e-2))
        δs_l1 = @views max(-T(1.5)*minimum(pt0.s_l[IntData.ilow]), T(1.e-4))
    end

    if IntData.n_upp == 0
        δx_u1, δs_u1 = zero(T), zero(T)
    else
        δx_u1 = max(-T(1.5)*minimum(uvar_m_x0), T(1.e-2))
        δs_u1 = @views max(-T(1.5)*minimum(pt0.s_u[IntData.iupp]), T(1.e-4))
    end

    x0_m_lvar .+= δx_l1
    uvar_m_x0 .+= δx_u1
    s0_l1 = @views pt0.s_l[IntData.ilow] .+ δs_l1
    s0_u1 = @views pt0.s_u[IntData.iupp] .+ δs_u1
    xs_l1, xs_u1 = s0_l1' * x0_m_lvar, s0_u1' * uvar_m_x0
    if IntData.n_low == 0
        δx_l2, δs_l2 = zero(T), zero(T)
    else
        δx_l2 = δx_l1 + xs_l1 / sum(s0_l1) / 2
        δs_l2 = @views δs_l1 + xs_l1 / sum(x0_m_lvar) / 2
    end
    if IntData.n_upp == 0
        δx_u2, δs_u2 = zero(T), zero(T)
    else
        δx_u2 = δx_u1 + xs_u1 / sum(s0_u1) / 2
        δs_u2 = @views δs_u1 + xs_u1 / sum(uvar_m_x0) / 2
    end
    δx = max(δx_l2, δx_u2)
    δs = max(δs_l2, δs_u2)
    pt0.x[IntData.ilow] .+= δx
    pt0.x[IntData.iupp] .-= δx
    pt0.s_l[IntData.ilow] = @views pt0.s_l[IntData.ilow] .+ δs
    pt0.s_u[IntData.iupp] = @views pt0.s_u[IntData.iupp] .+ δs

    @inbounds @simd for i in IntData.irng
        if FloatData.lvar[i] >= pt0.x[i]
            pt0.x[i] = FloatData.lvar[i] + T(1e-4)
        end
        if pt0.x[i] >= FloatData.uvar[i]
            pt0.x[i] = FloatData.uvar[i] - T(1e-4)
        end
        if (FloatData.lvar[i] < pt0.x[i] < FloatData.uvar[i]) == false
            pt0.x[i] = (FloatData.lvar[i] + FloatData.uvar[i]) / 2
        end
    end

    x0_m_lvar .= @views pt0.x[IntData.ilow] .- FloatData.lvar[IntData.ilow]
    uvar_m_x0 .= @views FloatData.uvar[IntData.iupp] .- pt0.x[IntData.iupp]

    @assert all(pt0.x .> FloatData.lvar) && all(pt0.x .< FloatData.uvar)
    @assert @views all(pt0.s_l[IntData.ilow] .> zero(T)) && all(pt0.s_u[IntData.iupp] .> zero(T))

    return pt0, J_fact, J_P, Qx, ATλ, x0_m_lvar, uvar_m_x0, Δ_xλ
end
