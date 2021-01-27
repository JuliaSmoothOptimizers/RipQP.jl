function get_norm_rc!(v, AT_colptr, AT_rowval, AT_nzval, n, ax)
    T = eltype(v)
    v .= zero(T)
    @inbounds @simd for j=1:n
        for i=AT_colptr[j] : (AT_colptr[j+1]-1)
            k = ax == :row ? AT_rowval[i] : j
            if abs(AT_nzval[i]) > v[k]
                v[k] = abs(AT_nzval[i])
            end
        end
    end

    v .= sqrt.(v)
    @inbounds @simd for i=1:length(v)
        if v[i] == zero(T)
            v[i] = one(T)
        end
    end
end

function mul_AT_D1_D2!(AT_colptr, AT_rowval, AT_nzval, d1, d2, r, c)
    @inbounds @simd for j=1:length(c)
        for i=AT_colptr[j]: (AT_colptr[j+1]-1)
            AT_nzval[i] /= r[AT_rowval[i]] * c[j]
        end
    end
    d1 ./= c
    d2 ./= r
end

function mul_AT_D3!(AT_colptr, AT_rowval, AT_nzval, n, d3)
    @inbounds @simd for j=1:n
        for i=AT_colptr[j]: (AT_colptr[j+1]-1)
            AT_nzval[i] *= d3[AT_rowval[i]]
        end
    end
end

function mul_Q_D!(Q_colptr, Q_rowval, Q_nzval, d, c)
    @inbounds @simd for j=1:length(d)
        for i=Q_colptr[j]: (Q_colptr[j+1]-1)
            Q_nzval[i] /= c[Q_rowval[i]] * c[j]
        end
    end
    d ./= c
end

function mul_Q_D2!(Q_colptr, Q_rowval, Q_nzval, d2)
    @inbounds @simd for j=1:length(d2)
        for i=Q_colptr[j]: (Q_colptr[j+1]-1)
            Q_nzval[i] *= d2[Q_rowval[i]] * d2[j]
        end
    end
end

function scaling_Ruiz!(FloatData_T0 :: QM_FloatData{T}, IntData :: QM_IntData, ϵ :: T;
                       max_iter :: Int = 100) where {T<:Real}
    d1, d2 = ones(T, IntData.n_rows), ones(T, IntData.n_cols)
    r_k, c_k = zeros(T, IntData.n_cols), zeros(T, IntData.n_rows)
    # r (resp. c) norm of rows of AT (resp. cols) 
    # scaling: D2 * AT * D1
    get_norm_rc!(r_k, FloatData_T0.AT.colptr, FloatData_T0.AT.rowval, FloatData_T0.AT.nzval, 
                 IntData.n_rows, :row)
    get_norm_rc!(c_k, FloatData_T0.AT.colptr, FloatData_T0.AT.rowval, FloatData_T0.AT.nzval, 
                 IntData.n_rows,:col)
    convergence = maximum(abs.(one(T) .- r_k)) <= ϵ && maximum(abs.(one(T) .- c_k)) <= ϵ
    mul_AT_D1_D2!(FloatData_T0.AT.colptr, FloatData_T0.AT.rowval, FloatData_T0.AT.nzval, d1, d2, r_k, c_k)
    k = 1
    while !convergence && k < max_iter
        get_norm_rc!(r_k, FloatData_T0.AT.colptr, FloatData_T0.AT.rowval, FloatData_T0.AT.nzval, 
                     IntData.n_rows, :row)
        get_norm_rc!(c_k, FloatData_T0.AT.colptr, FloatData_T0.AT.rowval, FloatData_T0.AT.nzval, 
                     IntData.n_rows,:col)
        convergence = maximum(abs.(one(T) .- r_k)) <= ϵ && maximum(abs.(one(T) .- c_k)) <= ϵ
        mul_AT_D1_D2!(FloatData_T0.AT.colptr, FloatData_T0.AT.rowval, FloatData_T0.AT.nzval, d1, d2, r_k, c_k)
        k += 1
    end

    mul_Q_D2!(FloatData_T0.Q.colptr, FloatData_T0.Q.rowval, FloatData_T0.Q.nzval, d2)
    FloatData_T0.b .*= d1
    FloatData_T0.c .*= d2
    FloatData_T0.lvar ./= d2
    FloatData_T0.uvar ./= d2

    # scaling Q (symmetric)
    d3 = ones(T, IntData.n_cols)
    r_k .= zero(T) # r_k is now norm of rows of Q
    get_norm_rc!(r_k, FloatData_T0.Q.colptr, FloatData_T0.Q.rowval, FloatData_T0.Q.nzval, 
                 IntData.n_cols,:row)
    convergence = maximum(abs.(one(T) .- r_k)) <= ϵ
    mul_Q_D!(FloatData_T0.Q.colptr, FloatData_T0.Q.rowval, FloatData_T0.Q.nzval, d3, r_k)
    k = 1
    while !convergence && k < max_iter
        get_norm_rc!(r_k, FloatData_T0.Q.colptr, FloatData_T0.Q.rowval, FloatData_T0.Q.nzval, 
                     IntData.n_cols,:row)
        convergence = maximum(abs.(one(T) .- r_k)) <= ϵ
        mul_Q_D!(FloatData_T0.Q.colptr, FloatData_T0.Q.rowval, FloatData_T0.Q.nzval, d3, r_k)
        k += 1
    end

    mul_AT_D3!(FloatData_T0.AT.colptr, FloatData_T0.AT.rowval, FloatData_T0.AT.nzval, FloatData_T0.AT.n, d3)
    FloatData_T0.c .*= d3
    FloatData_T0.lvar ./= d3
    FloatData_T0.uvar ./= d3

    return FloatData_T0, d1, d2, d3
end

function div_D2D3_Q_D3D2!(Q_colptr, Q_rowval, Q_nzval, d2, d3, n)
    @inbounds @simd for j=1:n
        for i=Q_colptr[j]: (Q_colptr[j+1]-1)
            Q_nzval[i] /= d2[Q_rowval[i]] * d2[j] * d3[Q_rowval[i]] * d3[j]
        end
    end 
end

function div_D1_A_D2D3!(AT_colptr, AT_rowval, AT_nzval, d1, d2, d3, n)
    @inbounds @simd for j=1:n
        for i= AT_colptr[j]: (AT_colptr[j+1]-1)
            AT_nzval[i] /= d1[j] * d2[AT_rowval[i]] * d3[AT_rowval[i]]
        end
    end
end

function post_scale(d1 :: Vector{T}, d2 :: Vector{T}, d3 :: Vector{T}, pt :: point{T}, res :: residuals{T},
                    FloatData_T0 :: QM_FloatData{T}, IntData :: QM_IntData, Qx :: Vector{T}, ATλ :: Vector{T},
                    Ax :: Vector{T}, cTx :: T, pri_obj :: T, dual_obj :: T, xTQx_2 :: T) where {T<:Real}
    pt.x .*= d2 .* d3
    div_D2D3_Q_D3D2!(FloatData_T0.Q.colptr, FloatData_T0.Q.rowval, FloatData_T0.Q.nzval, d2, d3, IntData.n_cols)
    Qx = mul!(Qx, Symmetric(FloatData_T0.Q, :U), pt.x)
    xTQx_2 =  pt.x' * Qx / 2
    div_D1_A_D2D3!(FloatData_T0.AT.colptr, FloatData_T0.AT.rowval, FloatData_T0.AT.nzval, d1, d2, d3, IntData.n_rows)
    pt.λ .*= d1
    ATλ = mul!(ATλ, FloatData_T0.AT, pt.λ)
    Ax = mul!(Ax, FloatData_T0.AT', pt.x)
    FloatData_T0.b ./= d1
    FloatData_T0.c ./= d2 .* d3
    cTx = FloatData_T0.c' * pt.x
    pri_obj = xTQx_2 + cTx + FloatData_T0.c0
    FloatData_T0.lvar .*= d2 .* d3
    FloatData_T0.uvar .*= d2 .* d3
    pt.s_l ./= d2 .* d3
    pt.s_u ./= d2 .* d3
    dual_obj = FloatData_T0.b' * pt.λ - xTQx_2 + view(pt.s_l, IntData.ilow)'*view(FloatData_T0.lvar, IntData.ilow) -
                    view(pt.s_u, IntData.iupp)'*view(FloatData_T0.uvar, IntData.iupp) + FloatData_T0.c0
    res.rb .= Ax .- FloatData_T0.b
    res.rc .= ATλ .-Qx .+ pt.s_l .- pt.s_u .- FloatData_T0.c
    #         rcNorm, rbNorm = norm(rc), norm(rb)
    res.rcNorm, res.rbNorm = norm(res.rc, Inf), norm(res.rb, Inf)

    return pt, pri_obj, res
end
