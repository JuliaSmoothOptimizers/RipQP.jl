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

function scaling_Ruiz!(FloatData_T0 :: QM_FloatData{T}, IntData :: QM_IntData, ϵ :: T;
                       max_iter :: Int = 100) where {T<:Real}
    n = length(IntData.Arows)
    d1, d2 = ones(T, IntData.n_rows), ones(T, IntData.n_cols)
    r_k, c_k = zeros(T, IntData.n_rows), zeros(T, IntData.n_cols)

    r_k = get_norm_rc!(r_k, IntData.Arows, FloatData_T0.Avals, IntData.n_rows, n)
    c_k = get_norm_rc!(c_k, IntData.Acols, FloatData_T0.Avals, IntData.n_cols, n)
    convergence = maximum(abs.(one(T) .- r_k)) <= ϵ && maximum(abs.(one(T) .- c_k)) <= ϵ
    IntData.Arows, IntData.Acols,
        FloatData_T0.Avals, d1, d2 = mul_A_D1_D2!(IntData.Arows, IntData.Acols, FloatData_T0.Avals,
                                                  d1, d2, r_k, c_k, IntData.n_rows, IntData.n_cols, n)
    k = 1
    while !convergence && k < max_iter
        r_k = get_norm_rc!(r_k, IntData.Arows, FloatData_T0.Avals, IntData.n_rows, n)
        c_k = get_norm_rc!(c_k, IntData.Acols, FloatData_T0.Avals, IntData.n_cols, n)
        convergence = maximum(abs.(one(T) .- r_k)) <= ϵ && maximum(abs.(one(T) .- c_k)) <= ϵ
        IntData.Arows, IntData.Acols,
            FloatData_T0.Avals, d1, d2 = mul_A_D1_D2!(IntData.Arows, IntData.Acols, FloatData_T0.Avals,
                                                      d1, d2, r_k, c_k, IntData.n_rows, IntData.n_cols, n)
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
    d3 = ones(T, IntData.n_cols)
    c_k .= zero(T)
    c_k = get_norm_rc!(c_k, IntData.Qcols, FloatData_T0.Qvals, IntData.n_cols, n_Q)
    convergence = maximum(abs.(one(T) .- c_k)) <= ϵ
    IntData.Qrows, IntData.Qcols,
        FloatData_T0.Qvals, d3 = mul_Q_D!(IntData.Qrows, IntData.Qcols, FloatData_T0.Qvals,
                                          d3, c_k, IntData.n_cols, n_Q)
    k = 1
    while !convergence && k < max_iter
        c_k = get_norm_rc!(c_k, IntData.Qcols, FloatData_T0.Qvals, IntData.n_cols, n_Q)
        convergence = maximum(abs.(one(T) .- c_k)) <= ϵ
        IntData.Qrows, IntData.Qcols,
            FloatData_T0.Qvals, d3 = mul_Q_D!(IntData.Qrows, IntData.Qcols, FloatData_T0.Qvals,
                                              d3, c_k, IntData.n_cols, n_Q)
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

function post_scale(d1 :: Vector{T}, d2 :: Vector{T}, d3 :: Vector{T}, pt :: point{T}, res :: residuals{T},
                    FloatData_T0 :: QM_FloatData{T}, IntData :: QM_IntData, Qx :: Vector{T}, ATλ :: Vector{T},
                    Ax :: Vector{T}, cTx :: T, pri_obj :: T, dual_obj :: T, xTQx_2 :: T) where {T<:Real}
    pt.x .*= d2 .* d3
    for i=1:length(IntData.Qrows)
        FloatData_T0.Qvals[i] /= d2[IntData.Qrows[i]] * d2[IntData.Qcols[i]] * d3[IntData.Qrows[i]] * d3[IntData.Qcols[i]]
    end
    Qx = mul_Qx_COO!(Qx, IntData.Qrows, IntData.Qcols, FloatData_T0.Qvals, pt.x)
    xTQx_2 =  pt.x' * Qx / 2
    for i=1:length(IntData.Arows)
        FloatData_T0.Avals[i] /= d1[IntData.Arows[i]] * d2[IntData.Acols[i]] * d3[IntData.Acols[i]]
    end
    pt.λ .*= d1
    ATλ = mul_ATλ_COO!(ATλ, IntData.Arows, IntData.Acols, FloatData_T0.Avals, pt.λ)
    Ax = mul_Ax_COO!(Ax, IntData.Arows, IntData.Acols, FloatData_T0.Avals, pt.x)
    FloatData_T0.b ./= d1
    FloatData_T0.c ./= d2 .* d3
    cTx = FloatData_T0.c' * pt.x
    pri_obj = xTQx_2 + cTx + FloatData_T0.c0
    FloatData_T0.lvar .*= d2 .* d3
    FloatData_T0.uvar .*= d2 .* d3
    dual_obj = FloatData_T0.b' * pt.λ - xTQx_2 + view(pt.s_l, IntData.ilow)'*view(FloatData_T0.lvar, IntData.ilow) -
                    view(pt.s_u, IntData.iupp)'*view(FloatData_T0.uvar, IntData.iupp) + FloatData_T0.c0
    pt.s_l ./= d2 .* d3
    pt.s_u ./= d2 .* d3
    res.rb .= Ax .- FloatData_T0.b
    res.rc .= ATλ .-Qx .+ pt.s_l .- pt.s_u .- FloatData_T0.c
    #         rcNorm, rbNorm = norm(rc), norm(rb)
    res.rcNorm, res.rbNorm = norm(res.rc, Inf), norm(res.rb, Inf)

    return pt, pri_obj, res
end
