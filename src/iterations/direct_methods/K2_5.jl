# Formulation K2: (if regul==:classic, adds additional regularization parmeters -ρ (top left) and δ (bottom right))
# [-Q_X - tmp_diag      sqrt(X1X2)A' ] [̃x] = rhs
# [ A sqrt(X1x2)                0    ] [y]
# where Q_X = sqrt(X1X2) Q sqrt(X1X2) and tmp_diag = s_l X2 + s_u X1
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
function factorize_K2_5!(J_augm, J_fact, tmp_diag, diag_Q , diagind_J, regu, s_l, s_u, x_m_lvar, uvar_m_x, 
                         ilow, iupp, n_rows, n_cols, cnts, qp, T, T0) 

    if regu.regul == :classic
        tmp_diag .= -regu.ρ
        J_augm.nzval[view(diagind_J, n_cols+1:n_rows+n_cols)] .= regu.δ
    else
        tmp_diag .= zero(T)
    end
    tmp_diag[ilow] .-= s_l ./ x_m_lvar
    tmp_diag[iupp] .-= s_u ./ uvar_m_x
    tmp_diag[diag_Q.nzind] .-= diag_Q.nzval
    J_augm.nzval[view(diagind_J,1:n_cols)] = tmp_diag 

    tmp_diag .= one(T)
    tmp_diag[ilow] .*= sqrt.(x_m_lvar)
    tmp_diag[iupp] .*= sqrt.(uvar_m_x)
    lrmultilply_J!(J_augm.colptr, J_augm.rowval, J_augm.nzval, tmp_diag, n_cols)

    if regu.regul == :dynamic
        # Amax = @views norm(J_augm.nzval[diagind_J], Inf)
        Amax = minimum(tmp_diag)
        if Amax < sqrt(eps(T)) && cnts.c_pdd < 8
            if T == Float32
                # restore J for next iteration
                tmp_diag .= one(T) 
                tmp_diag[ilow] ./= sqrt.(x_m_lvar)
                tmp_diag[iupp] ./= sqrt.(uvar_m_x)
                lrmultilply_J!(J_augm.colptr, J_augm.rowval, J_augm.nzval, tmp_diag, n_cols)
                return one(Int) # update to Float64
            elseif qp || cnts.c_pdd < 4
                cnts.c_pdd += 1
                regu.δ /= 10
                J_fact.r2 = max(sqrt(Amax), regu.δ)
                # regu.ρ /= 10
            end
        end
        J_fact.tol = min(Amax, T(eps(T)))
        J_fact = ldl_factorize!(Symmetric(J_augm, :U), J_fact)

    elseif regu.regul == :classic
        J_fact = try ldl_factorize!(Symmetric(J_augm, :U), J_fact)
        catch
            out = update_regu_trycatch!(regu, cnts, T, T0)
            if out == 1 
                # restore J for next iteration
                tmp_diag .= one(T) 
                tmp_diag[ilow] ./= sqrt.(x_m_lvar)
                tmp_diag[iupp] ./= sqrt.(uvar_m_x)
                lrmultilply_J!(J_augm.colptr, J_augm.rowval, J_augm.nzval, tmp_diag, n_cols)
                return out
            end
            cnts.c_catch += 1
            tmp_diag .= -regu.ρ       
            tmp_diag[ilow] .-= s_l ./ x_m_lvar
            tmp_diag[iupp] .-= s_u ./ uvar_m_x
            tmp_diag[diag_Q.nzind] .-= diag_Q.nzval
            J_augm.nzval[view(diagind_J,1:n_cols)] = tmp_diag 
        
            tmp_diag .= one(T)
            tmp_diag[ilow] .*= sqrt.(x_m_lvar)
            tmp_diag[iupp] .*= sqrt.(uvar_m_x)
            J_augm.nzval[view(diagind_J,1:n_cols)] .*= tmp_diag.^2
            J_augm.nzval[view(diagind_J, n_cols+1:n_rows+n_cols)] .= regu.δ
            J_fact = ldl_factorize!(Symmetric(J_augm, :U), J_fact)
        end

    else # no regularization
        J_fact = ldl_factorize!(Symmetric(J_augm, :U), J_fact)
    end

    cnts.c_catch >= 4 && return 1

    return 0 # factorization succeeded
end

function solve_augmented_system_aff_K2_5!(Δxy_aff, Δs_l_aff, Δs_u_aff, J_fact, rc, rb, x_m_lvar, uvar_m_x,
                                          s_l, s_u, tmp_diag, ilow, iupp, n_cols)

    Δxy_aff[1:n_cols] .= .-rc
    Δxy_aff[n_cols+1:end] .= .-rb
    Δxy_aff[ilow] .+= s_l
    Δxy_aff[iupp] .-= s_u
    Δxy_aff[1:n_cols] .*= tmp_diag

    ldiv!(J_fact, Δxy_aff)
    Δxy_aff[1:n_cols] .*= tmp_diag
    Δs_l_aff .= @views .-s_l .- s_l.*Δxy_aff[ilow]./x_m_lvar
    Δs_u_aff .= @views .-s_u .+ s_u.*Δxy_aff[iupp]./uvar_m_x
end

function solve_augmented_system_cc_K2_5!(J_fact, Δxy_cc, Δs_l_cc, Δs_u_cc, x_m_lvar, uvar_m_x, rxs_l, rxs_u, 
                                          s_l, s_u, tmp_diag, ilow, iupp, n_cols)

    Δxy_cc .= 0
    Δxy_cc[ilow] .+= rxs_l./x_m_lvar
    Δxy_cc[iupp] .+= rxs_u./uvar_m_x
    Δxy_cc[1:n_cols] .*= tmp_diag

    ldiv!(J_fact, Δxy_cc)
    Δxy_cc[1:n_cols] .*= tmp_diag
    Δs_l_cc .= @views .-(rxs_l.+s_l.*Δxy_cc[ilow])./x_m_lvar
    Δs_u_cc .= @views (rxs_u.+s_u.*Δxy_cc[iupp])./uvar_m_x
end

function solve_K2_5!(pt, itd, fd, id, res, pad, cnts, T, T0)

    out = factorize_K2_5!(itd.J_augm, itd.J_fact, itd.tmp_diag, itd.diag_Q, itd.diagind_J, itd.regu, 
                          pt.s_l, pt.s_u, itd.x_m_lvar, itd.uvar_m_x, id.ilow, id.iupp, 
                          id.n_rows, id.n_cols, cnts, itd.qp, T, T0)

    out == 1 && return out

    # affine scaling step
    solve_augmented_system_aff_K2_5!(pad.Δxy_aff, pad.Δs_l_aff, pad.Δs_u_aff, itd.J_fact, res.rc, res.rb, 
                                     itd.x_m_lvar, itd.uvar_m_x, pt.s_l, pt.s_u, itd.tmp_diag, 
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
    solve_augmented_system_cc_K2_5!(itd.J_fact, pad.Δxy, pad.Δs_l, pad.Δs_u, itd.x_m_lvar, itd.uvar_m_x, 
                                    pad.rxs_l, pad.rxs_u, pt.s_l, pt.s_u, itd.tmp_diag, id.ilow, id.iupp, 
                                    id.n_cols)

    # final direction
    pad.Δxy .+= pad.Δxy_aff  
    pad.Δs_l .+= pad.Δs_l_aff
    pad.Δs_u .+= pad.Δs_u_aff
    
    # update regularization
    if itd.regu.regul == :classic  # update ρ and δ values, check J_augm diag magnitude 
        out = update_regu_diagJ_K2_5!(itd.regu, itd.tmp_diag, itd.pdd, itd.l_pdd, itd.mean_pdd, cnts, T, T0) 
    end

    # restore J for next iteration
    itd.tmp_diag .= one(T) 
    itd.tmp_diag[id.ilow] ./= sqrt.(itd.x_m_lvar)
    itd.tmp_diag[id.iupp] ./= sqrt.(itd.uvar_m_x)
    lrmultilply_J!(itd.J_augm.colptr, itd.J_augm.rowval, itd.J_augm.nzval, itd.tmp_diag, id.n_cols)

    return out
end
