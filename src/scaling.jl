function get_norm_rc!(v, A_i, Avals, n_v, n)
    T = eltype(v)
    v .= zero(T)
    @inbounds @simd for j=1:n
        if abs(Avals[j]) > v[A_i[j]]
            v[A_i[j]] = abs(Avals[j])
        end
        #         v[A_i[j]] += Avals[j]^2  #2-norm
    end

    v = sqrt.(v)
    @inbounds @simd for i=1:n_v
        if v[i] == zero(T)
            v[i] = one(T)
        end
    end
    return v
end

function mul_A_D1_D2!(Arows, Acols, Avals, d1, d2, r, c, n_rows, n_cols, n)
    @inbounds @simd for i=1:n
        Avals[i] /= r[Arows[i]] * c[Acols[i]]
    end
    d1 ./= r
    d2 ./= c
    return Arows, Acols, Avals, d1, d2
end

function mul_Q_D!(Qrows, Qcols, Qvals, d, c, n_cols, n)
    @inbounds @simd for i=1:n
        Qvals[i] /= c[Qrows[i]] * c[Qcols[i]]
    end
    d ./= c
    return Qrows, Qcols, Qvals, d
end

function scaling_Ruiz!(FloatData_T0, IntData, n_rows, n_cols, ϵ; max_iter = 100)
    n = length(IntData.Arows)
    T = eltype(FloatData_T0.Avals)
    d1, d2 = ones(T, n_rows), ones(T, n_cols)
    r_k, c_k = zeros(T, n_rows), zeros(T, n_cols)

    r_k = get_norm_rc!(r_k, IntData.Arows, FloatData_T0.Avals, n_rows, n)
    c_k = get_norm_rc!(c_k, IntData.Acols, FloatData_T0.Avals, n_cols, n)
    convergence = maximum(abs.(one(T) .- r_k)) <= ϵ && maximum(abs.(one(T) .- c_k)) <= ϵ
    IntData.Arows, IntData.Acols,
        FloatData_T0.Avals, d1, d2 = mul_A_D1_D2!(IntData.Arows, IntData.Acols, FloatData_T0.Avals,
                                                  d1, d2, r_k, c_k, n_rows, n_cols, n)
    k = 1
    while !convergence && k < max_iter
        r_k = get_norm_rc!(r_k, IntData.Arows, FloatData_T0.Avals, n_rows, n)
        c_k = get_norm_rc!(c_k, IntData.Acols, FloatData_T0.Avals, n_cols, n)
        convergence = maximum(abs.(one(T) .- r_k)) <= ϵ && maximum(abs.(one(T) .- c_k)) <= ϵ
        IntData.Arows, IntData.Acols,
            FloatData_T0.Avals, d1, d2 = mul_A_D1_D2!(IntData.Arows, IntData.Acols, FloatData_T0.Avals,
                                                      d1, d2, r_k, c_k, n_rows, n_cols, n)
        k += 1
    end

    n_Q = length(IntData.Qrows)
    @inbounds @simd for i=1:n_Q
        FloatData_T0.Qvals[i] *= d2[IntData.Qrows[i]] * d2[IntData.Qcols[i]]
    end
    FloatData_T0.b .*= d1
    FloatData_T0.c .*= d2
    FloatData_T0.lvar ./= d2
    FloatData_T0.uvar ./= d2

    # scaling Q (symmetric)
    d3 = ones(T, n_cols)
    c_k .= zero(T)
    c_k = get_norm_rc!(c_k, IntData.Qcols, FloatData_T0.Qvals, n_cols, n_Q)
    convergence = maximum(abs.(one(T) .- c_k)) <= ϵ
    IntData.Qrows, IntData.Qcols,
        FloatData_T0.Qvals, d3 = mul_Q_D!(IntData.Qrows, IntData.Qcols, FloatData_T0.Qvals,
                                          d3, c_k, n_cols, n_Q)
    k = 1
    while !convergence && k < max_iter
        c_k = get_norm_rc!(c_k, IntData.Qcols, FloatData_T0.Qvals, n_cols, n_Q)
        convergence = maximum(abs.(one(T) .- c_k)) <= ϵ
        IntData.Qrows, IntData.Qcols,
            FloatData_T0.Qvals, d3 = mul_Q_D!(IntData.Qrows, IntData.Qcols, FloatData_T0.Qvals,
                                              d3, c_k, n_cols, n_Q)
        k += 1
    end

    for i=1:n
        FloatData_T0.Avals[i] *= d3[IntData.Acols[i]]
    end
    FloatData_T0.c .*= d3
    FloatData_T0.lvar ./= d3
    FloatData_T0.uvar ./= d3

    return FloatData_T0, d1, d2, d3
end

function post_scale(d1, d2, d3, x, λ, s_l, s_u, rb, rc, rcNorm, rbNorm, ilow, iupp,
                    FloatData_T0, IntData, Qx, ATλ, Ax, cTx, pri_obj, dual_obj, xTQx_2)
    x .*= d2 .* d3
    for i=1:length(IntData.Qrows)
        FloatData_T0.Qvals[i] /= d2[IntData.Qrows[i]] * d2[IntData.Qcols[i]] * d3[IntData.Qrows[i]] * d3[IntData.Qcols[i]]
    end
    Qx = mul_Qx_COO!(Qx, IntData.Qrows, IntData.Qcols, FloatData_T0.Qvals, x)
    xTQx_2 =  x' * Qx / 2
    for i=1:length(IntData.Arows)
        FloatData_T0.Avals[i] /= d1[IntData.Arows[i]] * d2[IntData.Acols[i]] * d3[IntData.Acols[i]]
    end
    λ .*= d1
    ATλ = mul_ATλ_COO!(ATλ, IntData.Arows, IntData.Acols, FloatData_T0.Avals, λ)
    Ax = mul_Ax_COO!(Ax, IntData.Arows, IntData.Acols, FloatData_T0.Avals, x)
    FloatData_T0.b ./= d1
    FloatData_T0.c ./= d2 .* d3
    cTx = FloatData_T0.c' * x
    pri_obj = xTQx_2 + cTx + FloatData_T0.c0
    FloatData_T0.lvar .*= d2 .* d3
    FloatData_T0.uvar .*= d2 .* d3
    dual_obj = FloatData_T0.b' * λ - xTQx_2 + view(s_l,ilow)'*view(FloatData_T0.lvar,ilow) -
                    view(s_u,iupp)'*view(FloatData_T0.uvar,iupp) + FloatData_T0.c0
    s_l ./= d2 .* d3
    s_u ./= d2 .* d3
    rb .= Ax .- FloatData_T0.b
    rc .= ATλ .-Qx .+ s_l .- s_u .- FloatData_T0.c
    #         rcNorm, rbNorm = norm(rc), norm(rb)
    rcNorm, rbNorm = norm(rc, Inf), norm(rb, Inf)

    return x, λ, s_l, s_u, pri_obj, rcNorm, rbNorm
end
