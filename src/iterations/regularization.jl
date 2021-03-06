# tools for the regularization of the system.
mutable struct regularization{T<:Real}
    ρ        :: T       # curent top-left regularization parameter
    δ        :: T       # cureent bottom-right regularization parameter
    ρ_min    :: T       # ρ minimum value 
    δ_min    :: T       # δ minimum value 
    regul    :: Symbol  # regularization mode (:classic, :dynamic, or :none)
end

convert(::Type{regularization{T}}, regu::regularization{T0}) where {T<:Real, T0<:Real} = 
    regularization(T(regu.ρ), T(regu.δ), T(regu.ρ_min), T(regu.δ_min), regu.regul)

# update regularization values in classic mode if there is a failure during factorization
function update_regu_trycatch!(regu, cnts, T, T0)
            
    T == Float32 && return 1
    T0 == Float128 && T == Float64 && return 1
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
    return 0
end

function update_regu!(regu) 
    if regu.δ >= regu.δ_min
        regu.δ /= 10
    end
    if regu.ρ >= regu.ρ_min
        regu.ρ /= 10
    end
end

# update regularization, and corrects if the magnitude of the diagonal of the matrix is too high
function update_regu_diagJ!(regu, J_augm_nzval, diagind_J, n_cols, pdd, l_pdd, mean_pdd, cnts, T, T0)

    l_pdd[cnts.k%6+1] = pdd
    mean_pdd = mean(l_pdd)

    if T == Float64 && cnts.k > 10  && mean_pdd!=zero(T) && std(l_pdd./mean_pdd) < T(1e-2) && cnts.c_pdd < 5
        regu.δ_min /= 10
        regu.δ /= 10
        cnts.c_pdd += 1
    end
    if T == Float64 && cnts.k>10 && cnts.c_catch <= 1 &&
            @views minimum(J_augm_nzval[diagind_J[1:n_cols]]) < -one(T) / regu.δ / T(1e-6)
        regu.δ /= 10
        regu.δ_min /= 10
        cnts.c_pdd += 1
    elseif T != T0 && cnts.c_pdd < 2 &&
            @views minimum(J_augm_nzval[diagind_J[1:n_cols]]) < -one(T) / regu.δ / T(1e-5)
        return 1
    elseif T == Float128 && cnts.k>10 && cnts.c_catch <= 1 &&
            @views minimum(J_augm_nzval[diagind_J[1:n_cols]]) < -one(T) / regu.δ / T(1e-15)
        regu.δ /= 10
        regu.δ_min /= 10
        cnts.c_pdd += 1
    end

    update_regu!(regu)

    return 0
end

function update_regu_diagJ_K2_5!(regu, tmp_diag, pdd, l_pdd, mean_pdd, cnts, T, T0)

    l_pdd[cnts.k%6+1] = pdd
    mean_pdd = mean(l_pdd)

    if T == Float64 && cnts.k > 10  && mean_pdd!=zero(T) && std(l_pdd./mean_pdd) < T(1e-2) && cnts.c_pdd < 5
        regu.δ_min /= 10
        regu.δ /= 10
        cnts.c_pdd += 1
    end
    if T == Float64 && cnts.k>10 && cnts.c_catch <= 1 &&
            @views minimum(tmp_diag) < -one(T) / regu.δ / T(1e-6)
        regu.δ /= 10
        regu.δ_min /= 10
        cnts.c_pdd += 1
    elseif T != T0 && cnts.c_pdd < 2 &&
            @views minimum(tmp_diag) < -one(T) / regu.δ / T(1e-5)
        return 1
    elseif T == Float128 && cnts.k>10 && cnts.c_catch <= 1 &&
            @views minimum(tmp_diag) < -one(T) / regu.δ / T(1e-15)
        regu.δ /= 10
        regu.δ_min /= 10
        cnts.c_pdd += 1
    end

    update_regu!(regu)

    return 0
end
