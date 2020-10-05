
function starting_points(FloatData, IntData, ilow, iupp, irng, J_augm , n_rows, n_cols, Δ_xλ)

    T = eltype(FloatData.Avals)
    J_P = ldl_analyze(Symmetric(J_augm, :U))
    J_fact = ldl_factorize!(Symmetric(J_augm, :U), J_P)
    Δ_xλ[n_cols+1: end] = FloatData.b
    Δ_xλ = ldiv!(J_fact, Δ_xλ)
    x0 = Δ_xλ[1:n_cols]
    λ0 = Δ_xλ[n_cols+1:end]
    s0_l, s0_u = zeros(T, n_cols), zeros(T, n_cols)
    Qx, ATλ = zeros(T, n_cols), zeros(T, n_cols)
    Qx = mul_Qx_COO!(Qx, IntData.Qrows, IntData.Qcols, FloatData.Qvals, x0)
    ATλ = mul_ATλ_COO!(ATλ, IntData.Arows, IntData.Acols, FloatData.Avals, λ0)
    dual_val = Qx - ATλ + FloatData.c
    s0_l[ilow] = @views dual_val[ilow]
    s0_u[iupp] = @views -dual_val[iupp]
    x0_m_lvar = @views x0[ilow] - FloatData.lvar[ilow]
    uvar_m_x0 = @views FloatData.uvar[iupp] - x0[iupp]
    if length(ilow) == 0
        δx_l1, δs_l1 = zero(T), zero(T)
    else
        δx_l1 = max(-T(1.5)*minimum(x0_m_lvar), T(1.e-2))
        δs_l1 = @views max(-T(1.5)*minimum(s0_l[ilow]), T(1.e-4))
    end

    if length(iupp) == 0
        δx_u1, δs_u1 = zero(T), zero(T)
    else
        δx_u1 = max(-T(1.5)*minimum(uvar_m_x0), T(1.e-2))
        δs_u1 = @views max(-T(1.5)*minimum(s0_u[iupp]), T(1.e-4))
    end

    x0_m_lvar .+= δx_l1
    uvar_m_x0 .+= δx_u1
    s0_l1 = @views s0_l[ilow] .+ δs_l1
    s0_u1 = @views s0_u[iupp] .+ δs_u1
    xs_l1, xs_u1 = s0_l1' * x0_m_lvar, s0_u1' * uvar_m_x0
    if length(ilow) == 0
        δx_l2, δs_l2 = zero(T), zero(T)
    else
        δx_l2 = δx_l1 + xs_l1 / sum(s0_l1) / 2
        δs_l2 = @views δs_l1 + xs_l1 / sum(x0_m_lvar) / 2
    end
    if length(iupp) == 0
        δx_u2, δs_u2 = zero(T), zero(T)
    else
        δx_u2 = δx_u1 + xs_u1 / sum(s0_u1) / 2
        δs_u2 = @views δs_u1 + xs_u1 / sum(uvar_m_x0) / 2
    end
    δx = max(δx_l2, δx_u2)
    δs = max(δs_l2, δs_u2)
    x0[ilow] .+= δx
    x0[iupp] .-= δx
    s0_l[ilow] = @views s0_l[ilow] .+ δs
    s0_u[iupp] = @views s0_u[iupp] .+ δs

    @inbounds @simd for i in irng
        if FloatData.lvar[i] >= x0[i]
            x0[i] = FloatData.lvar[i] + T(1e-4)
        end
        if x0[i] >= FloatData.uvar[i]
            x0[i] = FloatData.uvar[i] - T(1e-4)
        end
        if (FloatData.lvar[i] < x0[i] < FloatData.uvar[i]) == false
            x0[i] = (FloatData.lvar[i] + FloatData.uvar[i]) / 2
        end
    end

    x0_m_lvar .= @views x0[ilow] .- FloatData.lvar[ilow]
    uvar_m_x0 .= @views FloatData.uvar[iupp] .- x0[iupp]

    @assert all(x0 .> FloatData.lvar) && all(x0 .< FloatData.uvar)
    @assert @views all(s0_l[ilow] .> zero(T)) && all(s0_u[iupp] .> zero(T))

    return x0, λ0, s0_l, s0_u, J_fact, J_P, Qx, ATλ, x0_m_lvar, uvar_m_x0, Δ_xλ
end
