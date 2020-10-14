
function starting_points(FloatData, IntData, itd, Δ_xλ)

    T = eltype(FloatData.Avals)
    itd.J_P = ldl_analyze(Symmetric(itd.J_augm, :U))
    itd.J_fact = ldl_factorize!(Symmetric(itd.J_augm, :U), itd.J_P)
    Δ_xλ[IntData.n_cols+1: end] = FloatData.b
    Δ_xλ = ldiv!(itd.J_fact, Δ_xλ)
    pt0 = point(Δ_xλ[1:IntData.n_cols], Δ_xλ[IntData.n_cols+1:end], zeros(T, IntData.n_cols), zeros(T, IntData.n_cols))
    itd.Qx = mul_Qx_COO!(itd.Qx, IntData.Qrows, IntData.Qcols, FloatData.Qvals, pt0.x)
    itd.ATλ = mul_ATλ_COO!(itd.ATλ, IntData.Arows, IntData.Acols, FloatData.Avals, pt0.λ)
    dual_val = itd.Qx .- itd.ATλ .+ FloatData.c
    pt0.s_l[IntData.ilow] = @views dual_val[IntData.ilow]
    pt0.s_u[IntData.iupp] = @views -dual_val[IntData.iupp]
    itd.x_m_lvar .= @views pt0.x[IntData.ilow] .- FloatData.lvar[IntData.ilow]
    itd.uvar_m_x .= @views FloatData.uvar[IntData.iupp] .- pt0.x[IntData.iupp]
    if IntData.n_low == 0
        δx_l1, δs_l1 = zero(T), zero(T)
    else
        δx_l1 = max(-T(1.5)*minimum(itd.x_m_lvar), T(1.e-2))
        δs_l1 = @views max(-T(1.5)*minimum(pt0.s_l[IntData.ilow]), T(1.e-4))
    end

    if IntData.n_upp == 0
        δx_u1, δs_u1 = zero(T), zero(T)
    else
        δx_u1 = max(-T(1.5)*minimum(itd.uvar_m_x), T(1.e-2))
        δs_u1 = @views max(-T(1.5)*minimum(pt0.s_u[IntData.iupp]), T(1.e-4))
    end

    itd.x_m_lvar .+= δx_l1
    itd.uvar_m_x .+= δx_u1
    s0_l1 = @views pt0.s_l[IntData.ilow] .+ δs_l1
    s0_u1 = @views pt0.s_u[IntData.iupp] .+ δs_u1
    xs_l1, xs_u1 = s0_l1' * itd.x_m_lvar, s0_u1' * itd.uvar_m_x
    if IntData.n_low == 0
        δx_l2, δs_l2 = zero(T), zero(T)
    else
        δx_l2 = δx_l1 + xs_l1 / sum(s0_l1) / 2
        δs_l2 = @views δs_l1 + xs_l1 / sum(itd.x_m_lvar) / 2
    end
    if IntData.n_upp == 0
        δx_u2, δs_u2 = zero(T), zero(T)
    else
        δx_u2 = δx_u1 + xs_u1 / sum(s0_u1) / 2
        δs_u2 = @views δs_u1 + xs_u1 / sum(itd.uvar_m_x) / 2
    end
    δx = max(δx_l2, δx_u2)
    δs = max(δs_l2, δs_u2)
    pt0.x[IntData.ilow] .+= δx
    pt0.x[IntData.iupp] .-= δx
    pt0.s_l[IntData.ilow] .= @views pt0.s_l[IntData.ilow] .+ δs
    pt0.s_u[IntData.iupp] .= @views pt0.s_u[IntData.iupp] .+ δs

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

    itd.x_m_lvar .= @views pt0.x[IntData.ilow] .- FloatData.lvar[IntData.ilow]
    itd.uvar_m_x .= @views FloatData.uvar[IntData.iupp] .- pt0.x[IntData.iupp]

    @assert all(pt0.x .> FloatData.lvar) && all(pt0.x .< FloatData.uvar)
    @assert @views all(pt0.s_l[IntData.ilow] .> zero(T)) && all(pt0.s_u[IntData.iupp] .> zero(T))

    itd.Qx = mul_Qx_COO!(itd.Qx, IntData.Qrows, IntData.Qcols, FloatData.Qvals, pt0.x)
    itd.ATλ = mul_ATλ_COO!(itd.ATλ, IntData.Arows, IntData.Acols, FloatData.Avals, pt0.λ)
    itd.Ax = mul_Ax_COO!(itd.Ax, IntData.Arows, IntData.Acols, FloatData.Avals, pt0.x)
    itd.xTQx_2 = pt0.x' * itd.Qx / 2
    itd.cTx = FloatData.c' * pt0.x
    itd.pri_obj = itd.xTQx_2 + itd.cTx + FloatData.c0
    itd.dual_obj = FloatData.b' * pt0.λ - itd.xTQx_2 + view(pt0.s_l, IntData.ilow)'*view(FloatData.lvar, IntData.ilow) -
                    view(pt0.s_u, IntData.iupp)'*view(FloatData.uvar, IntData.iupp) + FloatData.c0
    itd.μ = @views compute_μ(itd.x_m_lvar, itd.uvar_m_x, pt0.s_l[IntData.ilow], pt0.s_u[IntData.iupp],
                             IntData.n_low, IntData.n_upp)
    itd.pdd = abs(itd.pri_obj - itd.dual_obj ) / (one(T) + abs(itd.pri_obj))

    return pt0, itd, Δ_xλ
end
