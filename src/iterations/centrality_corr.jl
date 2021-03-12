# Gondzio's multiple centrality correctors method

function update_rxs!(rxs_l, rxs_u, Hmin, Hmax, x_m_l_αΔp, u_m_x_αΔp, s_l_αΔp, s_u_αΔp, n_low, n_upp)
    
    @inbounds @simd for i=1:n_low
        rxs_l[i] = s_l_αΔp[i] * x_m_l_αΔp[i]
        if Hmin <= rxs_l[i] <= Hmax
            rxs_l[i] = 0
        elseif rxs_l[i] < Hmin
            rxs_l[i] -= Hmin
        else
            rxs_l[i] -= Hmax
        end
        if rxs_l[i] > Hmax
            rxs_l[i] = Hmax
        end
    end
    @inbounds @simd for i=1:n_upp
        rxs_u[i] = -s_u_αΔp[i]*u_m_x_αΔp[i]
        if Hmin <= -rxs_u[i] <= Hmax
            rxs_u[i] = 0
        elseif -rxs_u[i] < Hmin
            rxs_u[i] += Hmin
        else
            rxs_u[i] += Hmax
        end
        if rxs_u[i] < -Hmax
            rxs_u[i] = -Hmax
        end
    end
end

function centrality_corr!(Δxy_p, Δs_l_p, Δs_u_p, Δxy, Δs_l, Δs_u, α_p, α_d, K_fact, x, y, s_l, s_u, μ, 
                          rxs_l, rxs_u, lvar, uvar, x_m_lvar, uvar_m_x, x_m_l_αΔp, u_m_x_αΔp, s_l_αΔp, s_u_αΔp,
                          ilow, iupp, n_low, n_upp, n_rows, n_cols, corr_flag, iter_c, T) 
    # Δp = Δ_aff + Δ_cc
    δα, γ, βmin, βmax = T(0.1), T(0.1), T(0.1), T(10)
    α_p2, α_d2 = min(α_p + δα, one(T)), min(α_d + δα, one(T))
    update_pt_aff!(x_m_l_αΔp, u_m_x_αΔp, s_l_αΔp, s_u_αΔp, Δxy_p, Δs_l_p, Δs_u_p, x_m_lvar, uvar_m_x, 
                   s_l, s_u, α_p2, α_d2, ilow, iupp)
    μ_p = compute_μ(x_m_l_αΔp, u_m_x_αΔp, s_l_αΔp, s_u_αΔp, n_low, n_upp)

    σ = (μ_p / μ)^3
    Hmin, Hmax = βmin * σ * μ, βmax * σ * μ

    update_rxs!(rxs_l, rxs_u, Hmin, Hmax, x_m_l_αΔp, u_m_x_αΔp, s_l_αΔp, s_u_αΔp, n_low, n_upp)
    solve_augmented_system_cc!(K_fact, Δxy, Δs_l, Δs_u, x_m_lvar, uvar_m_x, rxs_l, rxs_u, s_l, s_u, ilow, iupp)
    
    Δxy .+= Δxy_p
    Δs_l .+= Δs_l_p 
    Δs_u .+= Δs_u_p
    α_p2, α_d2 = compute_αs(x, s_l, s_u, lvar, uvar, Δxy, Δs_l, Δs_u, n_cols)

    if α_p2 >= α_p + γ*δα && α_d2 >= α_d + γ*δα
        iter_c += 1
        Δxy_p .= Δxy
        Δs_l_p .= Δs_l
        Δs_u_p .= Δs_u
        α_p, α_d = α_p2, α_d2
    else
        Δxy .= Δxy_p
        Δs_l .= Δs_l_p
        Δs_u .= Δs_u_p
        corr_flag = false
    end

    return α_p, α_d, iter_c, corr_flag
end

function multi_centrality_corr!(pad, pt, α_pri, α_dual, K_fact, μ, lvar, uvar, x_m_lvar, uvar_m_x, id, kc, T)

    iter_c = 0 # current number of correction iterations
    corr_flag = true #stop correction if false
    # for storage issues Δ_aff = Δp  and Δ_cc = Δm
    pad.Δxy_aff .= pad.Δxy 
    pad.Δs_l_aff .= pad.Δs_l
    pad.Δs_u_aff .= pad.Δs_u
    @inbounds while iter_c < kc && corr_flag
        α_pri, α_dual, iter_c,
            corr_flag = centrality_corr!(pad.Δxy_aff, pad.Δs_l_aff, pad.Δs_u_aff, pad.Δxy, pad.Δs_l, pad.Δs_u, 
                                         α_pri, α_dual, K_fact, pt.x, pt.y, pt.s_l, pt.s_u, μ, pad.rxs_l, pad.rxs_u,
                                         lvar, uvar, x_m_lvar, uvar_m_x, pad.x_m_l_αΔ_aff,
                                         pad.u_m_x_αΔ_aff, pad.s_l_αΔ_aff, pad.s_u_αΔ_aff, id.ilow, id.iupp,
                                         id.n_low, id.n_upp, id.n_rows, id.n_cols, corr_flag, iter_c, T)
    end
    return α_pri, α_dual
end

# function to determine the number of centrality corrections (Gondzio's procedure)
function nb_corrector_steps(J_colptr, n_rows, n_cols, T) 
    Ef, Es, rfs = 0, 16 * n_cols, zero(T) # 14n = ratio tests and vector initializations
    @inbounds @simd for j=1:n_rows+n_cols
        lj = (J_colptr[j+1]-J_colptr[j])
        Ef += lj^2
        Es += lj
    end
    rfs = T(Ef / Es)
    if rfs <= 10
        kc = 0
    elseif 10 < rfs <= 30
        kc = 1
    elseif 30 < rfs <= 50
        kc = 2
    elseif rfs > 50
        kc = 3
    else
        p = Int(rfs / 50)
        kc = p + 2
        if kc > 10
            kc = 10
        end
    end
    return kc
end
