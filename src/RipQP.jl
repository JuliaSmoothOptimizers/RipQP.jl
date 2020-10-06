module RipQP

using LinearAlgebra, SparseArrays, Statistics

using LDLFactorizations, NLPModels, QuadraticModels, SolverTools

export ripqp

include("starting_points.jl")
include("scaling.jl")
include("sparse_toolbox.jl")
include("iterations.jl")
include("types_toolbox.jl")

function ripqp(QM0; mode = :mono, max_iter=800, ϵ_pdd=1e-8, ϵ_rb=1e-6, ϵ_rc=1e-6,
               tol_Δx=1e-16, ϵ_μ=1e-9, max_time=1200., scaling=true, display=true)

    if mode ∉ [:mono, :multi]
        error("mode should be :mono or :multi")
    end
    start_time = time()
    elapsed_time = 0.0
    QM = SlackModel(QM0)
    FloatData_T0, IntData, T = get_QM_data(QM)

    if scaling
        FloatData_T0, d1, d2, d3 = scaling_Ruiz!(FloatData_T0, IntData, T(1.0e-3))
    end
    # cNorm = norm(c)
    # bNorm = norm(b)
    # ANorm = norm(Avals) 
    # QNorm = norm(Qvals)

    if mode == :multi
        #change types
        T = Float32
        FloatData32, ϵ_pdd32, ϵ_rb32, ϵ_rc32,
            tol_Δx32, ϵ_μ32, ρ, δ, ρ_min, δ_min, tmp_diag,
            J_augm, diagind_J, diag_Q, x_m_l_αΔ_aff,
            u_m_x_αΔ_aff, s_l_αΔ_aff, s_u_αΔ_aff, rxs_l,
            rxs_u, Δ_aff, Δ_cc, Δ, Δ_xλ, x, λ, s_l, s_u,
            J_fact, J_P, Qx, ATλ, Ax, x_m_lvar, uvar_m_x,
            xTQx_2,  cTx, pri_obj, dual_obj, μ, pdd,
            rc, rb, rcNorm, rbNorm, tol_rb32, tol_rc32,
            tol_rb, tol_rc, optimal, small_Δx, small_μ,
            l_pdd, mean_pdd, n_Δx = init_params(T, FloatData_T0, IntData, tol_Δx, ϵ_μ, ϵ_rb, ϵ_rc)

    elseif mode == :mono
        # init regularization values
        ρ, δ, ρ_min, δ_min, tmp_diag, J_augm,
            diagind_J, diag_Q, x_m_l_αΔ_aff,
            u_m_x_αΔ_aff, s_l_αΔ_aff, s_u_αΔ_aff,
            rxs_l, rxs_u, Δ_aff, Δ_cc, Δ, Δ_xλ,
            x, λ, s_l, s_u, J_fact, J_P, Qx, ATλ, Ax,
            x_m_lvar, uvar_m_x, xTQx_2,  cTx,
            pri_obj, dual_obj, μ, pdd, rc, rb,
            rcNorm, rbNorm, tol_rb, tol_rc, optimal,
            small_Δx, small_μ,
            l_pdd, mean_pdd, n_Δx = init_params_mono(FloatData_T0, IntData, tol_Δx, ϵ_pdd, ϵ_μ, ϵ_rb, ϵ_rc)
    end

    Δt = time() - start_time
    tired = Δt > max_time
    k = 0
    c_catch = zero(Int) # to avoid endless loop
    c_pdd = zero(Int) # avoid too small δ_min

    # display
    if display == true
        @info log_header([:k, :pri_obj, :pdd, :rbNorm, :rcNorm, :n_Δx, :α_pri, :α_du, :μ],
        [Int, T, T, T, T, T, T, T, T, T],
        hdr_override=Dict(:k => "iter", :pri_obj => "obj", :pdd => "rgap",
        :rbNorm => "‖rb‖", :rcNorm => "‖rc‖",
        :n_Δx => "‖Δx‖"))
        @info log_row(Any[k, pri_obj, pdd, rbNorm, rcNorm, n_Δx, zero(T), zero(T), μ])
    end

    if mode == :multi
        # iters Float 32
        x, λ, s_l, s_u, x_m_lvar, uvar_m_x,
            rc, rb, rcNorm, rbNorm, Qx, ATλ,
            Ax, xTQx_2, cTx, pri_obj, dual_obj,
            pdd, l_pdd, mean_pdd, n_Δx, Δt,
            tired, optimal, μ, k, ρ, δ,
            ρ_min, δ_min, J_augm, J_fact,
            c_catch, c_pdd  = iter_mehrotraPC!(x, λ, s_l, s_u, x_m_lvar, uvar_m_x, FloatData32, IntData, rc, rb,
                                               rcNorm, rbNorm,
                                               tol_rb32, tol_rc32, Qx, ATλ, Ax, xTQx_2, cTx, pri_obj, dual_obj,
                                               pdd, l_pdd, mean_pdd, n_Δx, small_Δx, small_μ, Δt, tired, optimal,
                                               μ, k, ρ, δ, ρ_min, δ_min, J_augm, J_fact, J_P, diagind_J, diag_Q, tmp_diag,
                                               Δ_aff, Δ_cc, Δ, Δ_xλ, s_l_αΔ_aff, s_u_αΔ_aff, x_m_l_αΔ_aff, u_m_x_αΔ_aff,
                                               rxs_l, rxs_u, 30, ϵ_pdd32, ϵ_μ32, ϵ_rc32, ϵ_rb32, tol_Δx32,
                                               start_time, max_time, c_catch, c_pdd, display)

        # conversions to Float64
        T = Float64
        x, λ, s_l, s_u, x_m_lvar,
            uvar_m_x, rc, rb,
            rcNorm, rbNorm, Qx,
            ATλ, Ax, xTQx_2, cTx,
            pri_obj, dual_obj, pdd,
            l_pdd, mean_pdd, n_Δx,
            μ, ρ, δ, J_augm, J_P,
            J_fact, Δ_aff, Δ_cc, Δ,
            Δ_xλ, rxs_l, rxs_u, s_l_αΔ_aff,
            s_u_αΔ_aff, x_m_l_αΔ_aff,
            u_m_x_αΔ_aff, diag_Q,
            tmp_diag, ρ_min, δ_min = convert_types!(T, x, λ, s_l, s_u, x_m_lvar, uvar_m_x,
                                                    rc, rb,rcNorm, rbNorm, Qx, ATλ, Ax,
                                                    xTQx_2, cTx, pri_obj, dual_obj, pdd,
                                                    l_pdd, mean_pdd, n_Δx, μ, ρ, δ,
                                                    J_augm, J_P, J_fact, Δ_aff, Δ_cc, Δ,
                                                    Δ_xλ, rxs_l, rxs_u, s_l_αΔ_aff,
                                                    s_u_αΔ_aff, x_m_l_αΔ_aff, u_m_x_αΔ_aff,
                                                    diag_Q, tmp_diag, ρ_min, δ_min)

        optimal = pdd < ϵ_pdd && rbNorm < tol_rb && rcNorm < tol_rc
        small_Δx, small_μ = n_Δx < tol_Δx, μ < ϵ_μ
        ρ /= 10
        δ /= 10
        optimal = pdd < ϵ_pdd && rbNorm < tol_rb && rcNorm < tol_rc
    end

    # iters T0
    x, λ, s_l, s_u, x_m_lvar, uvar_m_x,
        rc, rb, rcNorm, rbNorm, Qx, ATλ,
        Ax, xTQx_2, cTx, pri_obj, dual_obj,
        pdd, l_pdd, mean_pdd, n_Δx, Δt,
        tired, optimal, μ, k, ρ, δ,
        ρ_min, δ_min, J_augm, J_fact,
        c_catch, c_pdd  = iter_mehrotraPC!(x, λ, s_l, s_u, x_m_lvar, uvar_m_x, FloatData_T0, IntData,
                                           rc, rb, rcNorm, rbNorm,
                                           tol_rb, tol_rc, Qx, ATλ, Ax, xTQx_2, cTx, pri_obj, dual_obj,
                                           pdd, l_pdd, mean_pdd, n_Δx, small_Δx, small_μ, Δt, tired, optimal,
                                           μ, k, ρ, δ, ρ_min, δ_min, J_augm, J_fact, J_P, diagind_J, diag_Q,
                                           tmp_diag, Δ_aff, Δ_cc, Δ, Δ_xλ, s_l_αΔ_aff, s_u_αΔ_aff,
                                           x_m_l_αΔ_aff, u_m_x_αΔ_aff, rxs_l, rxs_u, max_iter, ϵ_pdd, ϵ_μ, ϵ_rc, ϵ_rb,
                                           tol_Δx, start_time, max_time, c_catch, c_pdd, display)

    if k>= max_iter
        status = :max_iter
    elseif tired
        status = :max_time
    elseif optimal
        status = :acceptable
    else
        status = :unknown
    end

    if scaling
        x, λ, s_l, s_u, pri_obj,
            rcNorm, rbNorm = post_scale(d1, d2, d3, x, λ, s_l, s_u, rb, rc, rcNorm,
                                        rbNorm, FloatData_T0, IntData,
                                        Qx, ATλ, Ax, cTx, pri_obj, dual_obj, xTQx_2)
    end

    elapsed_time = time() - start_time

    stats = GenericExecutionStats(status, QM, solution = x[1:QM.meta.nvar],
                                  objective = pri_obj,
                                  dual_feas = rcNorm,
                                  primal_feas = rbNorm,
                                  multipliers = λ,
                                  multipliers_L = s_l,
                                  multipliers_U = s_u,
                                  iter = k,
                                  elapsed_time=elapsed_time)
    return stats
end

end
