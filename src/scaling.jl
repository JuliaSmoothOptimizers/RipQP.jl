function get_norm_rc!(v, A, ax)
    T = eltype(v)
    v .= zero(T)
    @inbounds @simd for j=1:size(A, 2)
        for i in nzrange(A, j)
            k = ax == :row ? A.rowval[i] : j
            if abs(A.nzval[i]) > v[k]
                v[k] = abs(A.nzval[i])
            end
        end
    end

    v = sqrt.(v)
    @inbounds @simd for i=1:length(v)
        if v[i] == zero(T)
            v[i] = one(T)
        end
    end
    return v
end

function mul_A_D1_D2!(A, d1, d2, r, c)
    @inbounds @simd for j=1:size(A, 2)
        for i in nzrange(A, j)
            A.nzval[i] /= r[A.rowval[i]] * c[j]
        end
    end
    d1 ./= r
    d2 ./= c
    return A, d1, d2
end

function mul_Q_D!(Q, d, c)
    @inbounds @simd for j=1:size(Q, 2)
        for i in nzrange(Q, j)
            Q.nzval[i] /= c[Q.rowval[i]] * c[j]
        end
    end
    d ./= c
    return Q, d
end

function scaling_Ruiz!(FloatData_T0 :: QM_FloatData{T}, IntData :: QM_IntData, ϵ :: T;
                       max_iter :: Int = 100) where {T<:Real}
    d1, d2 = ones(T, IntData.n_rows), ones(T, IntData.n_cols)
    r_k, c_k = zeros(T, IntData.n_rows), zeros(T, IntData.n_cols)

    r_k = get_norm_rc!(r_k, FloatData_T0.A, :row)
    c_k = get_norm_rc!(c_k,  FloatData_T0.A, :col)
    convergence = maximum(abs.(one(T) .- r_k)) <= ϵ && maximum(abs.(one(T) .- c_k)) <= ϵ
    FloatData_T0.A, d1, d2 = mul_A_D1_D2!(FloatData_T0.A, d1, d2, r_k, c_k)
    k = 1
    while !convergence && k < max_iter
        r_k = get_norm_rc!(r_k, FloatData_T0.A, :row)
        c_k = get_norm_rc!(c_k,  FloatData_T0.A, :col)
        convergence = maximum(abs.(one(T) .- r_k)) <= ϵ && maximum(abs.(one(T) .- c_k)) <= ϵ
        FloatData_T0.A, d1, d2 = mul_A_D1_D2!(FloatData_T0.A, d1, d2, r_k, c_k)
        k += 1
    end

    @inbounds @simd for j=1:size(FloatData_T0.Q, 2)
        for i in nzrange(FloatData_T0.Q, j)
            FloatData_T0.Q.nzval[i] *= d2[FloatData_T0.Q.rowval[i]] * d2[j]
        end
    end
    FloatData_T0.b .*= d1
    FloatData_T0.c .*= d2
    FloatData_T0.lvar ./= d2
    FloatData_T0.uvar ./= d2

    # scaling Q (symmetric)
    d3 = ones(T, IntData.n_cols)
    c_k .= zero(T)
    c_k = get_norm_rc!(c_k, FloatData_T0.Q, :col)
    convergence = maximum(abs.(one(T) .- c_k)) <= ϵ
    FloatData_T0.Q, d3 = mul_Q_D!(FloatData_T0.Q, d3, c_k)
    k = 1
    while !convergence && k < max_iter
        c_k = get_norm_rc!(c_k, FloatData_T0.Q, :col)
        convergence = maximum(abs.(one(T) .- c_k)) <= ϵ
        FloatData_T0.Q, d3 = mul_Q_D!(FloatData_T0.Q, d3, c_k)
        k += 1
    end

    @inbounds @simd for j=1:size(FloatData_T0.A, 2)
        for i in nzrange(FloatData_T0.A, j)
            FloatData_T0.A.nzval[i] *= d3[j]
        end
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
    @inbounds @simd for j=1:size(FloatData_T0.Q, 2)
        for i in nzrange(FloatData_T0.Q, j)
            FloatData_T0.Q.nzval[i] /= d2[FloatData_T0.Q.rowval[i]] * d2[j] * d3[FloatData_T0.Q.rowval[i]] * d3[j]
        end
    end
    Qx = mul!(Qx, Symmetric(FloatData_T0.Q, :L), pt.x)
    xTQx_2 =  pt.x' * Qx / 2
    @inbounds @simd for j=1:size(FloatData_T0.A, 2)
        for i in nzrange(FloatData_T0.A, j)
            FloatData_T0.A.nzval[i] /= d1[FloatData_T0.A.rowval[i]] * d2[j] * d3[j]
        end
    end
    pt.λ .*= d1
    ATλ = mul!(ATλ, FloatData_T0.A', pt.λ)
    Ax = mul!(Ax, FloatData_T0.A, pt.x)
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
