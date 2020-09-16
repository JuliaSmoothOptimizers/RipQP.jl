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

function scaling_Ruiz!(Arows, Acols, Avals, Qrows, Qcols, Qvals, c, b, lvar, uvar,
                       n_rows, n_cols, ϵ; max_iter = 100)
    n = length(Arows)
    T = eltype(Avals)
    d1, d2 = ones(T, n_rows), ones(T, n_cols)
    r_k, c_k = zeros(T, n_rows), zeros(T, n_cols)

    r_k = get_norm_rc!(r_k, Arows, Avals, n_rows, n)
    c_k = get_norm_rc!(c_k, Acols, Avals, n_cols, n)
    convergence = maximum(abs.(one(T) .- r_k)) <= ϵ && maximum(abs.(one(T) .- c_k)) <= ϵ
    Arows, Acols, Avals, d1, d2 = mul_A_D1_D2!(Arows, Acols, Avals, d1, d2,
                                               r_k, c_k, n_rows, n_cols, n)
    k = 1
    while !convergence && k < max_iter
        r_k = get_norm_rc!(r_k, Arows, Avals, n_rows, n)
        c_k = get_norm_rc!(c_k, Acols, Avals, n_cols, n)
        convergence = maximum(abs.(one(T) .- r_k)) <= ϵ && maximum(abs.(one(T) .- c_k)) <= ϵ
        Arows, Acols, Avals, d1, d2 = mul_A_D1_D2!(Arows, Acols, Avals, d1, d2,
                                                   r_k, c_k, n_rows, n_cols, n)
        k += 1
    end

    n_Q = length(Qrows)
    @inbounds @simd for i=1:n_Q
        Qvals[i] *= d2[Qrows[i]] * d2[Qcols[i]]
    end
    b .*= d1
    c .*= d2
    lvar ./= d2
    uvar ./= d2

    # scaling Q (symmetric)
    d3 = ones(T, n_cols)
    c_k .= zero(T)
    c_k = get_norm_rc!(c_k, Qcols, Qvals, n_cols, n_Q)
    convergence = maximum(abs.(one(T) .- c_k)) <= ϵ
    Qrows, Qcols, Qvals, d3 = mul_Q_D!(Qrows, Qcols, Qvals, d3, c_k, n_cols, n_Q)
    k = 1
    while !convergence && k < max_iter
        c_k = get_norm_rc!(c_k, Qcols, Qvals, n_cols, n_Q)
        convergence = maximum(abs.(one(T) .- c_k)) <= ϵ
        Qrows, Qcols, Qvals, d3 = mul_Q_D!(Qrows, Qcols, Qvals, d3, c_k, n_cols, n_Q)
        k += 1
    end

    for i=1:n
        Avals[i] *= d3[Acols[i]]
    end
    c .*= d3
    lvar ./= d3
    uvar ./= d3

    return Arows, Acols, Avals, Qrows, Qcols, Qvals, c, b, lvar, uvar, d1, d2, d3
end

function post_scale(d1, d2, d3, x, λ, s_l, s_u, rb, rc, rcNorm, rbNorm, lvar, uvar,
                    ilow, iupp, b, c, c0, Qrows, Qcols, Qvals, Arows, Acols, Avals,
                    Qx, ATλ, Ax, cTx, pri_obj, dual_obj, xTQx_2)
    x .*= d2 .* d3
    for i=1:length(Qrows)
        Qvals[i] /= d2[Qrows[i]] * d2[Qcols[i]] * d3[Qrows[i]] * d3[Qcols[i]]
    end
    Qx = mul_Qx_COO!(Qx, Qrows, Qcols, Qvals, x)
    xTQx_2 =  x' * Qx / 2
    for i=1:length(Arows)
        Avals[i] /= d1[Arows[i]] * d2[Acols[i]] * d3[Acols[i]]
    end
    λ .*= d1
    ATλ = mul_ATλ_COO!(ATλ, Arows, Acols, Avals, λ)
    Ax = mul_Ax_COO!(Ax, Arows, Acols, Avals, x)
    b ./= d1
    c ./= d2 .* d3
    cTx = c' * x
    pri_obj = xTQx_2 + cTx + c0
    lvar .*= d2 .* d3
    uvar .*= d2 .* d3
    dual_obj = b' * λ - xTQx_2 + view(s_l,ilow)'*view(lvar,ilow) -
                    view(s_u,iupp)'*view(uvar,iupp) +c0
    s_l ./= d2 .* d3
    s_u ./= d2 .* d3
    rb .= Ax .- b
    rc .= ATλ .-Qx .+ s_l .- s_u .- c
    #         rcNorm, rbNorm = norm(rc), norm(rb)
    rcNorm, rbNorm = norm(rc, Inf), norm(rb, Inf)

    return x, λ, s_l, s_u, pri_obj, rcNorm, rbNorm
end
