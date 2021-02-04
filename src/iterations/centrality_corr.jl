# Gondzio's multiple centrality correctors method

function update_rxs!(rxs_l :: Vector{T}, rxs_u :: Vector{T}, Hmin :: T, Hmax :: T, 
                     x_m_l_αΔp :: Vector{T}, u_m_x_αΔp :: Vector{T}, s_l_αΔp :: Vector{T}, 
                     s_u_αΔp :: Vector{T}, n_low :: Int, n_upp :: Int) where {T<:Real}

    @inbounds @simd for i=1:n_low
        rxs_l[i] = s_l_αΔp[i] * x_m_l_αΔp[i]
        if Hmin <= rxs_l[i] <= Hmax
            rxs_l[i] = zero(T)
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
            rxs_u[i] = zero(T)
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

function centrality_corr!(Δp :: Vector{T}, Δm :: Vector{T}, Δ :: Vector{T}, Δ_xy :: Vector{T}, 
                          α_p :: T, α_d :: T, J_fact :: LDLFactorizations.LDLFactorization{T,Int,Int,Int} , 
                          x :: Vector{T}, y :: Vector{T}, s_l :: Vector{T}, s_u :: Vector{T}, μ :: T, 
                          rxs_l :: Vector{T}, rxs_u :: Vector{T}, lvar :: Vector{T}, uvar :: Vector{T}, 
                          x_m_lvar :: Vector{T}, uvar_m_x :: Vector{T}, x_m_l_αΔp :: Vector{T}, 
                          u_m_x_αΔp :: Vector{T}, s_l_αΔp :: Vector{T}, s_u_αΔp :: Vector{T},
                          ilow :: Vector{Int}, iupp :: Vector{Int}, n_low :: Int, n_upp :: Int, n_rows :: Int, 
                          n_cols :: Int, corr_flag :: Bool, k_corr :: Int) where {T<:Real}
    # Δp = Δ_aff + Δ_cc
    δα, γ, βmin, βmax = T(0.1), T(0.1), T(0.1), T(10)
    α_p2, α_d2 = min(α_p + δα, one(T)), min(α_d + δα, one(T))

    update_pt_aff!(x_m_l_αΔp, u_m_x_αΔp, s_l_αΔp, s_u_αΔp, Δp, x_m_lvar, uvar_m_x, s_l, s_u, α_p2, α_d2, 
                   ilow, iupp, n_low, n_rows, n_cols)
    μ_p = compute_μ(x_m_l_αΔp, u_m_x_αΔp, s_l_αΔp, s_u_αΔp, n_low, n_upp)

    σ = (μ_p / μ)^3
    Hmin, Hmax = βmin * σ * μ, βmax * σ * μ

    update_rxs!(rxs_l, rxs_u, Hmin, Hmax, x_m_l_αΔp, u_m_x_αΔp, s_l_αΔp, s_u_αΔp, n_low, n_upp)
    solve_augmented_system_cc!(Δm, J_fact, Δ_xy , Δp, x_m_lvar, uvar_m_x, rxs_l, rxs_u, s_l, s_u,
                               ilow, iupp, n_cols, n_rows, n_low)
    Δ .= Δp .+ Δm
    α_p2, α_d2 = compute_αs(x, s_l, s_u, lvar, uvar, Δ, ilow, iupp, n_low, n_upp, n_rows, n_cols)

    if α_p2 >= α_p + γ*δα && α_d2 >= α_d + γ*δα
        k_corr += 1
        Δp .= Δ
        α_p, α_d = α_p2, α_d2
    else
        Δ .= Δp
        corr_flag = false
    end

    return Δp, Δ, α_p, α_d, k_corr, corr_flag
end

function multi_centrality_corr!(pad :: preallocated_data{T}, pt :: point{T}, α_pri :: T, α_dual :: T,
                                J_fact :: LDLFactorizations.LDLFactorization{T,Int,Int,Int}, μ :: T, 
                                lvar :: Vector{T}, uvar :: Vector{T}, x_m_lvar :: Vector{T}, 
                                uvar_m_x :: Vector{T}, id :: QM_IntData, K :: Int) where {T<:Real}

    k_corr = 0
    corr_flag = true #stop correction if false
    pad.Δ_aff .= pad.Δ # for storage issues Δ_aff = Δp  and Δ_cc = Δm
    @inbounds while k_corr < K && corr_flag
        pad.Δ_aff, pad.Δ, α_pri, α_dual, k_corr,
            corr_flag = centrality_corr!(pad.Δ_aff, pad.Δ_cc, pad.Δ, pad.Δ_xy, α_pri, α_dual,
                                         J_fact, pt.x, pt.y, pt.s_l, pt.s_u, μ, pad.rxs_l, pad.rxs_u,
                                         lvar, uvar, x_m_lvar, uvar_m_x, pad.x_m_l_αΔ_aff,
                                         pad.u_m_x_αΔ_aff, pad.s_l_αΔ_aff, pad.s_u_αΔ_aff, id.ilow, id.iupp,
                                         id.n_low, id.n_upp, id.n_rows, id.n_cols, corr_flag, k_corr)
    end
    return α_pri, α_dual
end