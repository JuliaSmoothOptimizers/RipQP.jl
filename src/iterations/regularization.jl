# tools for the regularization of the system.

# update regularization values in classic mode if there is a failure during factorization
function update_regu_trycatch!(regu :: regularization{T}, cnts :: counters, T0 :: DataType) where {T<:Real}
            
    T == Float32 && return one(Int)
    T0 == Float128 && T == Float64 && return one(Int)
    if cnts.c_pdd == 0 && cnts.c_catch == 0
        regu.δ *= T(1e2)
        regu.δ_min *= T(1e2)
        regu.ρ *= T(1e5)
        regu.ρ_min *= T(1e5)
    elseif cnts.c_pdd == 0 && cnts.c_catch != 0
        regu.δ *= T(1e1)
        regu.δ_min *= T(1e1)
        regu.ρ *= T(1e0)
        regu.ρ_min *= T(1e0)
    elseif cnts.c_pdd != 0 && cnts.c_catch==0
        regu.δ *= T(1e5)
        regu.δ_min *= T(1e5)
        regu.ρ *= T(1e5)
        regu.ρ_min *= T(1e5)
    else
        regu.δ *= T(1e1)
        regu.δ_min *= T(1e1)
        regu.ρ *= T(1e1)
        regu.ρ_min *= T(1e1)
    end
    return zero(Int)
end

function update_regu!(regu :: regularization{T}) where {T<:Real}
    if regu.δ >= regu.δ_min
        regu.δ /= 10
    end
    if regu.ρ >= regu.ρ_min
        regu.ρ /= 10
    end
end

# update regularization, and corrects if the magnitude of the diagonal of the matrix is too high
function update_regu_diagJ!(regu :: regularization{T}, J_augm_nzval :: Vector{T}, diagind_J :: Vector{Int},
                            n_cols :: Int, pdd :: T, l_pdd :: Vector{T}, mean_pdd :: T, cnts :: counters, 
                            T0 :: DataType) where {T<:Real}

    l_pdd[cnts.k%6+1] = pdd
    mean_pdd = mean(l_pdd)

    if T == Float64 && cnts.k > 10  && mean_pdd!=zero(T) && std(l_pdd./mean_pdd) < 1e-2 && cnts.c_pdd < 5
        regu.δ_min /= 10
        regu.δ /= 10
        cnts.c_pdd += 1
    end
    if T == Float64 && cnts.k>10 && cnts.c_catch <= 1 &&
            @views minimum(J_augm_nzval[view(diagind_J,1:n_cols)]) < -one(T) / regu.δ / T(1e-6)
        regu.δ /= 10
        regu.δ_min /= 10
        cnts.c_pdd += 1
    elseif T != T0 && cnts.c_pdd < 2 &&
            @views minimum(J_augm_nzval[view(diagind_J,1:n_cols)]) < -one(T) / regu.δ / T(1e-5)
        return one(Int)
    elseif T == Float128 && cnts.k>10 && cnts.c_catch <= 1 &&
            @views minimum(J_augm_nzval[view(diagind_J,1:n_cols)]) < -one(T) / regu.δ / T(1e-15)
        regu.δ /= 10
        regu.δ_min /= 10
        cnts.c_pdd += 1
    end

    update_regu!(regu)

    return zero(Int)
end