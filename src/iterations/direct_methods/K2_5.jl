# Formulation K2: (if regul==:classic, adds additional regularization parmeters -ρ (top left) and δ (bottom right))
# [-Q_X - D      sqrt(X1X2)A' ] [̃x] = rhs
# [ A sqrt(X1x2)                0    ] [y]
# where Q_X = sqrt(X1X2) Q sqrt(X1X2) and D = s_l X2 + s_u X1
# and  Δ x = sqrt(X1 X2) Δ ̃x
 
# iter_data type for the K2 formulation

# we init data with the same function as in create_iterdata_K2, so we do not create this function 
# for the K2.5 formulation.

function lrmultilply_J!(J_colptr, J_rowval, J_nzval, v, n_cols)
    T = eltype(v)
    @inbounds @simd for i=1:n_cols
        for idx_row=J_colptr[i]: J_colptr[i+1]-1
                J_nzval[idx_row] *= v[i] * v[J_rowval[idx_row]]
        end
    end

    n = length(J_colptr)
    @inbounds @simd for i=n_cols+1:n-1
        for idx_row= J_colptr[i]: J_colptr[i+1] - 1
            if J_rowval[idx_row] <= n_cols
                J_nzval[idx_row] *= v[J_rowval[idx_row]] # multiply row i by v[i]
            end
        end
    end
end

# function lrmultiply_J2!(J, v)
#     lmul!(v, J)
#     rmul!(J, v)
# end


# iteration functions for the K2.5 system
function factorize_K2_5!(K, K_fact, D, diag_Q , diagind_K, regu, s_l, s_u, x_m_lvar, uvar_m_x, 
                         ilow, iupp, n_rows, n_cols, cnts, qp, T, T0) 

    if regu.regul == :classic
        D .= -regu.ρ
        K.nzval[view(diagind_K, n_cols+1:n_rows+n_cols)] .= regu.δ
    else
        D .= zero(T)
    end
    D[ilow] .-= s_l ./ x_m_lvar
    D[iupp] .-= s_u ./ uvar_m_x
    D[diag_Q.nzind] .-= diag_Q.nzval
    K.nzval[view(diagind_K,1:n_cols)] = D 

    D .= one(T)
    D[ilow] .*= sqrt.(x_m_lvar)
    D[iupp] .*= sqrt.(uvar_m_x)
    lrmultilply_J!(K.colptr, K.rowval, K.nzval, D, n_cols)

    if regu.regul == :dynamic
        # Amax = @views norm(K.nzval[diagind_K], Inf)
        Amax = minimum(D)
        if Amax < sqrt(eps(T)) && cnts.c_pdd < 8
            if T == Float32
                # restore J for next iteration
                D .= one(T) 
                D[ilow] ./= sqrt.(x_m_lvar)
                D[iupp] ./= sqrt.(uvar_m_x)
                lrmultilply_J!(K.colptr, K.rowval, K.nzval, D, n_cols)
                return one(Int) # update to Float64
            elseif qp || cnts.c_pdd < 4
                cnts.c_pdd += 1
                regu.δ /= 10
                K_fact.r2 = max(sqrt(Amax), regu.δ)
                # regu.ρ /= 10
            end
        end
        K_fact.tol = min(Amax, T(eps(T)))
        K_fact = ldl_factorize!(Symmetric(K, :U), K_fact)

    elseif regu.regul == :classic
        K_fact = try ldl_factorize!(Symmetric(K, :U), K_fact)
        catch
            out = update_regu_trycatch!(regu, cnts, T, T0)
            if out == 1 
                # restore J for next iteration
                D .= one(T) 
                D[ilow] ./= sqrt.(x_m_lvar)
                D[iupp] ./= sqrt.(uvar_m_x)
                lrmultilply_J!(K.colptr, K.rowval, K.nzval, D, n_cols)
                return out
            end
            cnts.c_catch += 1
            D .= -regu.ρ       
            D[ilow] .-= s_l ./ x_m_lvar
            D[iupp] .-= s_u ./ uvar_m_x
            D[diag_Q.nzind] .-= diag_Q.nzval
            K.nzval[view(diagind_K,1:n_cols)] = D 
        
            D .= one(T)
            D[ilow] .*= sqrt.(x_m_lvar)
            D[iupp] .*= sqrt.(uvar_m_x)
            K.nzval[view(diagind_K,1:n_cols)] .*= D.^2
            K.nzval[view(diagind_K, n_cols+1:n_rows+n_cols)] .= regu.δ
            K_fact = ldl_factorize!(Symmetric(K, :U), K_fact)
        end

    else # no regularization
        K_fact = ldl_factorize!(Symmetric(K, :U), K_fact)
    end

    cnts.c_catch >= 4 && return 1

    return 0 # factorization succeeded
end

function solve_augmented_system_aff_K2_5!(Δxy_aff, Δs_l_aff, Δs_u_aff, K_fact, rc, rb, x_m_lvar, uvar_m_x,
                                          s_l, s_u, D, ilow, iupp, n_cols)

    Δxy_aff[1:n_cols] .= .-rc
    Δxy_aff[n_cols+1:end] .= .-rb
    Δxy_aff[ilow] .+= s_l
    Δxy_aff[iupp] .-= s_u
    Δxy_aff[1:n_cols] .*= D

    ldiv!(K_fact, Δxy_aff)
    Δxy_aff[1:n_cols] .*= D
    Δs_l_aff .= @views .-s_l .- s_l.*Δxy_aff[ilow]./x_m_lvar
    Δs_u_aff .= @views .-s_u .+ s_u.*Δxy_aff[iupp]./uvar_m_x
end

function solve_augmented_system_cc_K2_5!(K_fact, Δxy_cc, Δs_l_cc, Δs_u_cc, x_m_lvar, uvar_m_x, rxs_l, rxs_u, 
                                          s_l, s_u, D, ilow, iupp, n_cols)

    Δxy_cc .= 0
    Δxy_cc[ilow] .+= rxs_l./x_m_lvar
    Δxy_cc[iupp] .+= rxs_u./uvar_m_x
    Δxy_cc[1:n_cols] .*= D

    ldiv!(K_fact, Δxy_cc)
    Δxy_cc[1:n_cols] .*= D
    Δs_l_cc .= @views .-(rxs_l.+s_l.*Δxy_cc[ilow])./x_m_lvar
    Δs_u_cc .= @views (rxs_u.+s_u.*Δxy_cc[iupp])./uvar_m_x
end

function solve_K2_5!(pt, itd, fd, id, res, pad, cnts, T, T0)

    out = factorize_K2_5!(pad.K, pad.K_fact, itd.D, pad.diag_Q, pad.diagind_K, itd.regu, 
                          pt.s_l, pt.s_u, itd.x_m_lvar, itd.uvar_m_x, id.ilow, id.iupp, 
                          id.n_rows, id.n_cols, cnts, itd.qp, T, T0)

    out == 1 && return out

    # affine scaling step
    solve_augmented_system_aff_K2_5!(pad.Δxy_aff, pad.Δs_l_aff, pad.Δs_u_aff, pad.K_fact, res.rc, res.rb, 
                                     itd.x_m_lvar, itd.uvar_m_x, pt.s_l, pt.s_u, itd.D, 
                                     id.ilow, id.iupp, id.n_cols)
    α_aff_pri, α_aff_dual = compute_αs(pt.x, pt.s_l, pt.s_u, fd.lvar, fd.uvar, pad.Δxy_aff, pad.Δs_l_aff, 
                                       pad.Δs_u_aff, id.n_cols)
    # (x-lvar, uvar-x, s_l, s_u) .+= α_aff * Δ_aff                                 
    update_pt_aff!(pad.x_m_l_αΔ_aff, pad.u_m_x_αΔ_aff, pad.s_l_αΔ_aff, pad.s_u_αΔ_aff, pad.Δxy_aff, 
                   pad.Δs_l_aff, pad.Δs_u_aff, itd.x_m_lvar, itd.uvar_m_x, pt.s_l, pt.s_u, 
                   α_aff_pri, α_aff_dual, id.ilow, id.iupp)
    μ_aff = compute_μ(pad.x_m_l_αΔ_aff, pad.u_m_x_αΔ_aff, pad.s_l_αΔ_aff, pad.s_u_αΔ_aff, id.n_low, id.n_upp)
    σ = (μ_aff / itd.μ)^3

    # corrector-centering step
    pad.rxs_l .= @views -σ * itd.μ .+ pad.Δxy_aff[id.ilow] .* pad.Δs_l_aff
    pad.rxs_u .= @views σ * itd.μ .+ pad.Δxy_aff[id.iupp] .* pad.Δs_u_aff
    solve_augmented_system_cc_K2_5!(pad.K_fact, pad.Δxy, pad.Δs_l, pad.Δs_u, itd.x_m_lvar, itd.uvar_m_x, 
                                    pad.rxs_l, pad.rxs_u, pt.s_l, pt.s_u, itd.D, id.ilow, id.iupp, 
                                    id.n_cols)

    # final direction
    pad.Δxy .+= pad.Δxy_aff  
    pad.Δs_l .+= pad.Δs_l_aff
    pad.Δs_u .+= pad.Δs_u_aff
    
    # update regularization
    if itd.regu.regul == :classic  # update ρ and δ values, check K diag magnitude 
        out = update_regu_diagJ_K2_5!(itd.regu, itd.D, itd.pdd, itd.l_pdd, itd.mean_pdd, cnts, T, T0) 
    end

    # restore J for next iteration
    itd.D .= one(T) 
    itd.D[id.ilow] ./= sqrt.(itd.x_m_lvar)
    itd.D[id.iupp] ./= sqrt.(itd.uvar_m_x)
    lrmultilply_J!(pad.K.colptr, pad.K.rowval, pad.K.nzval, itd.D, id.n_cols)

    return out
end
