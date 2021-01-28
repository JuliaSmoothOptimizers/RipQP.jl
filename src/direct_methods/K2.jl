# Formulation K2:
# [-Q - tmp_diag     A' ] [x] = rhs
# [ A                0  ] [y]

# Init functions for the K2 system
function fill_J_augm_K2!(J_augm_colptr, J_augm_rowval, J_augm_nzval, tmp_diag, Q_colptr, Q_rowval, Q_nzval,
                         AT_colptr, AT_rowval, AT_nzval, diag_Q_nzind, δ, n_rows, n_cols, regul, T)

    added_coeffs_diag = 0 # we add coefficients that do not appear in Q in position i,i if Q[i,i] = 0
    n_nz = length(diag_Q_nzind)
    c_nz = n_nz > 0 ? 1 : 0
    @inbounds for j=1:n_cols  # Q coeffs, tmp diag coefs. 
        J_augm_colptr[j+1] = Q_colptr[j+1] + added_coeffs_diag 
        for k=Q_colptr[j]:(Q_colptr[j+1]-2)
            nz_idx = k + added_coeffs_diag 
            J_augm_rowval[nz_idx] = Q_rowval[k]
            J_augm_nzval[nz_idx] = -Q_nzval[k]
        end
        if c_nz == 0 || c_nz > n_nz || diag_Q_nzind[c_nz] != j
            added_coeffs_diag += 1
            J_augm_colptr[j+1] += 1
            nz_idx = J_augm_colptr[j+1] - 1
            J_augm_rowval[nz_idx] = j
            J_augm_nzval[nz_idx] = tmp_diag[j]
        else
            if c_nz != 0  
                c_nz += 1
            end
            nz_idx = J_augm_colptr[j+1] - 1
            J_augm_rowval[nz_idx] = j
            J_augm_nzval[nz_idx] = tmp_diag[j] - Q_nzval[Q_colptr[j+1]-1] 
        end
    end

    countsum = J_augm_colptr[n_cols+1] # current value of J_augm_colptr[Q.n+j+1]
    nnz_top_left = countsum # number of coefficients + 1 already added
    @inbounds for j=1:n_rows
        countsum += AT_colptr[j+1] - AT_colptr[j] 
        if regul == :classic 
            countsum += 1
        end
        J_augm_colptr[n_cols+j+1] = countsum
        for k=AT_colptr[j]:(AT_colptr[j+1]-1)
            nz_idx = regul == :classic ? k + nnz_top_left + j - 2 : k + nnz_top_left - 1
            J_augm_rowval[nz_idx] = AT_rowval[k]
            J_augm_nzval[nz_idx] = AT_nzval[k]
        end
        if regul == :classic
            nz_idx = J_augm_colptr[n_cols+j+1] - 1
            J_augm_rowval[nz_idx] = n_cols + j 
            J_augm_nzval[nz_idx] = δ
        end
    end   
end

function create_J_augm_K2(id, tmp_diag, Q, AT, diag_Q, regu, T)
    # for classic regul only
    n_nz = length(tmp_diag) - length(diag_Q.nzind) + length(AT.nzval) + length(Q.nzval) 
    if regu.regul == :classic
        n_nz += id.n_rows
    end
    J_augm_colptr = Vector{Int}(undef, id.n_rows+id.n_cols+1) 
    J_augm_colptr[1] = 1
    J_augm_rowval = Vector{Int}(undef, n_nz)
    J_augm_nzval = Vector{T}(undef, n_nz)
    # [-Q -tmp_diag    AT]
    # [0               δI]

    fill_J_augm_K2!(J_augm_colptr, J_augm_rowval, J_augm_nzval, tmp_diag, Q.colptr, Q.rowval, Q.nzval,
                    AT.colptr, AT.rowval, AT.nzval, diag_Q.nzind, regu.δ, id.n_rows, 
                    id.n_cols, regu.regul, T)

    return SparseMatrixCSC(id.n_rows+id.n_cols, id.n_rows+id.n_cols,
                           J_augm_colptr, J_augm_rowval, J_augm_nzval)
end

# iteration functions for the K2 system
function J_update_factorize_K2!(J_augm, J_fact, tmp_diag, diag_Q, diagind_J, regu, 
                               s_l, s_u, x_m_lvar, uvar_m_x, ilow, iupp, n_rows, n_cols,
                               cnts)

    if regu.regul == :classic
        tmp_diag .= -regu.ρ
        J_augm.nzval[view(diagind_J, n_cols+1:n_rows+n_cols)] .= regu.δ
    else
        tmp_diag .= zero(T)
    end
    tmp_diag[ilow] .-= @views s_l[ilow] ./ x_m_lvar
    tmp_diag[iupp] .-= @views s_u[iupp] ./ uvar_m_x
    tmp_diag[diag_Q.nzind] .-= diag_Q.nzval
    J_augm.nzval[view(diagind_J,1:n_cols)] = tmp_diag 

    if regu.regul == :dynamic
        Amax = @views norm(J_augm.nzval[itd.diagind_J], Inf)
        if Amax > T(1e6) / regu.δ && cnts.c_pdd < 8
            if T == Float32
                return one(Int) # update to Float64
            elseif length(fQ.nzval) > 0 || cnts.c_pdd < 4
                cnts.c_pdd += 1
                regu.δ /= 10
                # regu.ρ /= 10
            end
        end
        J_fact = ldl_factorize!(Symmetric(J_augm, :U), J_fact,  tol=Amax*T(eps(T)), 
                                r1=-regu.ρ, r2=regu.δ, n_d=n_cols)

    elseif regu.regul == :classic
        J_fact = try ldl_factorize!(Symmetric(J_augm, :U), J_fact)
        catch
            if T == Float32
                return one(Int)
            elseif T0 == Float128 && T == Float64
                return one(Int)
            end
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
            cnts.c_catch += 1
            tmp_diag .= -regu.ρ
            tmp_diag[ilow] .-= @views s_l[ilow] ./ x_m_lvar
            tmp_diag[iupp] .-= @views s_u[iupp] ./ uvar_m_x
            tmp_diag[diag_Q.nzind] .-= diag_Q.nzval
            J_augm.nzval[view(diagind_J,1:n_cols)] = tmp_diag 
            J_augm.nzval[view(diagind_J, n_cols+1:n_rows+n_cols)] .= regu.δ
            J_fact = ldl_factorize!(Symmetric(J_augm, :U), J_fact)
        end

    else # no regularization
        itd.J_fact = ldl_factorize!(Symmetric(itd.J_augm, :U), itd.J_fact)
    end

    if cnts.c_catch >= 4
        return one(Int)
    end

    return zero(Int) # factorization succeeded
end

function solve_K2!(pt :: point{T}, itd :: iter_data{T}, fd :: QM_FloatData{T}, id :: QM_IntData,
                   res :: residuals{T}, regu :: regularization{T}, pad :: preallocated_data{T},
                   cnts :: counters) where {T:<Real}

    out = J_update_factorize_K2!(itd.J_augm, itd.J_fact, itd.tmp_diag, itd.diag_Q, itd.diagind_J, regu, 
                                 pt.s_l, pt.s_u, itd.x_m_lvar, itd.uvar_m_x, id.ilow, id.iupp, 
                                 id.n_rows, id.n_cols, cnts)

    if out == one(T) 
        return out
    end

    # affine scaling step
    pad.Δ_aff = solve_augmented_system_aff!(itd.J_fact, pad.Δ_aff, pad.Δ_xy, res.rc, res.rb,
                                            itd.x_m_lvar, itd.uvar_m_x, pt.s_l, pt.s_u,
                                            id.ilow, id.iupp, id.n_cols, id.n_rows,
                                            id.n_low)
    α_aff_pri = @views compute_α_primal(pt.x, pad.Δ_aff[1:id.n_cols], fd.lvar, fd.uvar)
    α_aff_dual_l = @views compute_α_dual(pt.s_l[id.ilow], pad.Δ_aff[id.n_rows+id.n_cols+1:id.n_rows+id.n_cols+id.n_low])
    α_aff_dual_u = @views compute_α_dual(pt.s_u[id.iupp], pad.Δ_aff[id.n_rows+id.n_cols+id.n_low+1:end])
    # alpha_aff_dual is the min of the 2 alpha_aff_dual
    α_aff_dual = min(α_aff_dual_l, α_aff_dual_u)
    pad.x_m_l_αΔ_aff .= @views itd.x_m_lvar .+ α_aff_pri .* pad.Δ_aff[1:id.n_cols][id.ilow]
    pad.u_m_x_αΔ_aff .= @views itd.uvar_m_x .- α_aff_pri .* pad.Δ_aff[1:id.n_cols][id.iupp]
    pad.s_l_αΔ_aff .= @views pt.s_l[id.ilow] .+ α_aff_dual .* pad.Δ_aff[id.n_rows+id.n_cols+1: id.n_rows+id.n_cols+id.n_low]
    pad.s_u_αΔ_aff .= @views pt.s_u[id.iupp] .+ α_aff_dual .* pad.Δ_aff[id.n_rows+id.n_cols+id.n_low+1: end]
    μ_aff = compute_μ(pad.x_m_l_αΔ_aff, pad.u_m_x_αΔ_aff, pad.s_l_αΔ_aff, pad.s_u_αΔ_aff, id.n_low, id.n_upp)
    σ = (μ_aff / itd.μ)^3

    # corrector-centering step
    pad.Δ_cc = solve_augmented_system_cc!(itd.J_fact, pad.Δ_cc, pad.Δ_xy , pad.Δ_aff, σ, itd.μ,
                                          itd.x_m_lvar, itd.uvar_m_x, pad.rxs_l, pad.rxs_u, pt.s_l, pt.s_u,
                                          id.ilow, id.iupp, id.n_cols, id.n_rows,
                                          id.n_low)
    return out
end
