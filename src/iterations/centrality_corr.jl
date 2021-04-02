# Gondzio's multiple centrality correctors method

function update_rxs!(rxs_l, rxs_u, Hmin, Hmax, x_m_l_αΔp, u_m_x_αΔp, s_l_αΔp, s_u_αΔp, nlow, nupp)
    
    @inbounds @simd for i=1:nlow
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
    @inbounds @simd for i=1:nupp
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

function multi_centrality_corr!(dda :: DescentDirectionAllocsPC{T}, pad :: PreallocatedData{T}, pt :: Point{T}, α_pri :: T, α_dual :: T, 
                                itd :: IterData{T}, fd :: Abstract_QM_FloatData{T}, id :: QM_IntData, cnts :: Counters, res :: Residuals{T}, 
                                T0 :: DataType) where {T<:Real}

    iter_c = 0 # current number of correction iterations
    corr_flag = true #stop correction if false
    # for storage issues Δ_aff = Δp  and Δ_cc = Δm
    dda.Δxy_aff .= itd.Δxy 
    dda.Δs_l_aff .= itd.Δs_l
    dda.Δs_u_aff .= itd.Δs_u
    @inbounds while iter_c < cnts.kc && corr_flag
        # Δp = Δ_aff + Δ_cc
        δα, γ, βmin, βmax = T(0.2), T(0.1), T(0.1), T(10)
        α_p2, α_d2 = min(α_pri + δα, one(T)), min(α_dual + δα, one(T))
        update_pt_aff!(dda.x_m_l_αΔ_aff, dda.u_m_x_αΔ_aff, dda.s_l_αΔ_aff, dda.s_u_αΔ_aff, dda.Δxy_aff, dda.Δs_l_aff, dda.Δs_u_aff, 
                        itd.x_m_lvar, itd.uvar_m_x, pt.s_l, pt.s_u, α_p2, α_d2, id.ilow, id.iupp)
        μ_p = compute_μ(dda.x_m_l_αΔ_aff, dda.u_m_x_αΔ_aff, dda.s_l_αΔ_aff, dda.s_u_αΔ_aff, id.nlow, id.nupp)

        σ = (μ_p / itd.μ)^3
        Hmin, Hmax = βmin * σ * itd.μ, βmax * σ * itd.μ

        # corrector-centering step
        update_rxs!(dda.rxs_l, dda.rxs_u, Hmin, Hmax, dda.x_m_l_αΔ_aff, dda.u_m_x_αΔ_aff, dda.s_l_αΔ_aff, dda.s_u_αΔ_aff, id.nlow, id.nupp)
        itd.Δxy .= 0
        itd.Δxy[id.ilow] .+= dda.rxs_l ./ itd.x_m_lvar
        itd.Δxy[id.iupp] .+= dda.rxs_u ./ itd.uvar_m_x
        out = solver!(pad, dda, pt, itd, fd, id, res, cnts, T0, :cc)
        itd.Δs_l .= @views .-(dda.rxs_l .+ pt.s_l .* itd.Δxy[id.ilow]) ./ itd.x_m_lvar
        itd.Δs_u .= @views (dda.rxs_u .+ pt.s_u .* itd.Δxy[id.iupp]) ./ itd.uvar_m_x
        
        itd.Δxy .+= dda.Δxy_aff
        itd.Δs_l .+= dda.Δs_l_aff 
        itd.Δs_u .+= dda.Δs_u_aff
        α_p2, α_d2 = compute_αs(pt.x, pt.s_l, pt.s_u, fd.lvar, fd.uvar, itd.Δxy, itd.Δs_l, itd.Δs_u, id.nvar)

        if α_p2 >= α_pri + γ*δα && α_d2 >= α_dual + γ*δα
            iter_c += 1
            dda.Δxy_aff .= itd.Δxy
            dda.Δs_l_aff .= itd.Δs_l
            dda.Δs_u_aff .= itd.Δs_u
            α_pri, α_dual = α_p2, α_d2
        else
            itd.Δxy .= dda.Δxy_aff
            itd.Δs_l .= dda.Δs_l_aff
            itd.Δs_u .= dda.Δs_u_aff
            corr_flag = false
        end
    end
    return α_pri, α_dual
end

# function to determine the number of centrality corrections (Gondzio's procedure)
function nb_corrector_steps(J_colptr, ncon, nvar, T) 
    Ef, Es, rfs = 0, 16 * nvar, zero(T) # 14n = ratio tests and vector initializations
    @inbounds @simd for j=1:ncon+nvar
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

function nb_corrector_steps!(cnts :: Counters, time_fact :: T, time_solve :: T) where {T<:Real}
    rfs = time_fact / time_solve
    if rfs <= T(10)
        cnts.kc = 0
    elseif T(10) < rfs <= T(30)
        cnts.kc = 1
    elseif T(30) < rfs <= T(50)
        cnts.kc = 2
    elseif rfs > T(50)
        cnts.kc = 3
    else
        p = Int(round(rfs / 50))
        cntd.kc = p + 2
        if cnts.kc > 10
            cnts.kc = 10
        end
    end
end
