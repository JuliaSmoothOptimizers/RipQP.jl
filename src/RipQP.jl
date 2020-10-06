module RipQP

using LinearAlgebra, SparseArrays, Statistics

using LDLFactorizations, NLPModels, QuadraticModels, SolverTools

export ripqp

include("starting_points.jl")
include("scaling.jl")
include("sparse_toolbox.jl")
include("iterations.jl")
include("types_toolbox.jl")
include("types_definition.jl")

function ripqp(QM0; mode = :mono, max_iter=800, ϵ_pdd=1e-8, ϵ_rb=1e-6, ϵ_rc=1e-6,
               ϵ_Δx=1e-16, ϵ_μ=1e-9, max_time=1200., scaling=true, display=true)

    if mode ∉ [:mono, :multi]
        error("mode should be :mono or :multi")
    end
    start_time = time()
    elapsed_time = 0.0
    QM = SlackModel(QM0)
    FloatData_T0, IntData, T = get_QM_data(QM)
    ϵ = tolerances(ϵ_pdd, ϵ_rb, ϵ_rc, ϵ_μ, ϵ_Δx)

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
        FloatData32, ϵ32, regu, tmp_diag,
            J_augm, diagind_J, diag_Q, x_m_l_αΔ_aff,
            u_m_x_αΔ_aff, s_l_αΔ_aff, s_u_αΔ_aff, rxs_l,
            rxs_u, Δ_aff, Δ_cc, Δ, Δ_xλ, pt,
            J_fact, J_P, Qx, ATλ, Ax, x_m_lvar, uvar_m_x,
            xTQx_2,  cTx, pri_obj, dual_obj, μ, pdd,
            res, tol_rb32, tol_rc32,
            tol_rb, tol_rc, optimal, small_Δx, small_μ,
            l_pdd, mean_pdd, n_Δx = init_params(T, FloatData_T0, IntData, ϵ.μ, ϵ.Δx)

    elseif mode == :mono
        # init regularization values
        regu, tmp_diag, J_augm,
            diagind_J, diag_Q, x_m_l_αΔ_aff,
            u_m_x_αΔ_aff, s_l_αΔ_aff, s_u_αΔ_aff,
            rxs_l, rxs_u, Δ_aff, Δ_cc, Δ, Δ_xλ,
            pt, J_fact, J_P, Qx, ATλ, Ax,
            x_m_lvar, uvar_m_x, xTQx_2,  cTx,
            pri_obj, dual_obj, μ, pdd, res, tol_rb, tol_rc, optimal,
            small_Δx, small_μ,
            l_pdd, mean_pdd, n_Δx = init_params_mono(FloatData_T0, IntData, ϵ)
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
        @info log_row(Any[k, pri_obj, pdd, res.rbNorm, res.rcNorm, n_Δx, zero(T), zero(T), μ])
    end

    if mode == :multi
        # iters Float 32
        pt, x_m_lvar, uvar_m_x,
            res, Qx, ATλ,
            Ax, xTQx_2, cTx, pri_obj, dual_obj,
            pdd, l_pdd, mean_pdd, n_Δx, Δt,
            tired, optimal, μ, k, regu,
            J_augm, J_fact,
            c_catch, c_pdd  = iter_mehrotraPC!(pt, x_m_lvar, uvar_m_x, FloatData32, IntData, res,
                                               tol_rb32, tol_rc32, Qx, ATλ, Ax, xTQx_2, cTx, pri_obj, dual_obj,
                                               pdd, l_pdd, mean_pdd, n_Δx, small_Δx, small_μ, Δt, tired, optimal,
                                               μ, k, regu, J_augm, J_fact, J_P, diagind_J, diag_Q, tmp_diag,
                                               Δ_aff, Δ_cc, Δ, Δ_xλ, s_l_αΔ_aff, s_u_αΔ_aff, x_m_l_αΔ_aff, u_m_x_αΔ_aff,
                                               rxs_l, rxs_u, 30, ϵ32,
                                               start_time, max_time, c_catch, c_pdd, display)

        # conversions to Float64
        T = Float64
        pt, x_m_lvar,
            uvar_m_x, res, Qx,
            ATλ, Ax, xTQx_2, cTx,
            pri_obj, dual_obj, pdd,
            l_pdd, mean_pdd, n_Δx,
            μ, regu, J_augm, J_P,
            J_fact, Δ_aff, Δ_cc, Δ,
            Δ_xλ, rxs_l, rxs_u, s_l_αΔ_aff,
            s_u_αΔ_aff, x_m_l_αΔ_aff,
            u_m_x_αΔ_aff, diag_Q,
            tmp_diag,  = convert_types!(T, pt, x_m_lvar, uvar_m_x,
                                                    res, Qx, ATλ, Ax,
                                                    xTQx_2, cTx, pri_obj, dual_obj, pdd,
                                                    l_pdd, mean_pdd, n_Δx, μ, regu,
                                                    J_augm, J_P, J_fact, Δ_aff, Δ_cc, Δ,
                                                    Δ_xλ, rxs_l, rxs_u, s_l_αΔ_aff,
                                                    s_u_αΔ_aff, x_m_l_αΔ_aff, u_m_x_αΔ_aff,
                                                    diag_Q, tmp_diag)

        optimal = pdd < ϵ_pdd && res.rbNorm < tol_rb && res.rcNorm < tol_rc
        small_Δx, small_μ = n_Δx < ϵ_Δx, μ < ϵ_μ
        regu.ρ /= 10
        regu.δ /= 10
        optimal = pdd < ϵ_pdd && res.rbNorm < tol_rb && res.rcNorm < tol_rc
    end

    # iters T0
    pt, x_m_lvar, uvar_m_x,
        res, Qx, ATλ,
        Ax, xTQx_2, cTx, pri_obj, dual_obj,
        pdd, l_pdd, mean_pdd, n_Δx, Δt,
        tired, optimal, μ, k,
        regu, J_augm, J_fact,
        c_catch, c_pdd  = iter_mehrotraPC!(pt, x_m_lvar, uvar_m_x, FloatData_T0, IntData, res,
                                           tol_rb, tol_rc, Qx, ATλ, Ax, xTQx_2, cTx, pri_obj, dual_obj,
                                           pdd, l_pdd, mean_pdd, n_Δx, small_Δx, small_μ, Δt, tired, optimal,
                                           μ, k, regu, J_augm, J_fact, J_P, diagind_J, diag_Q,
                                           tmp_diag, Δ_aff, Δ_cc, Δ, Δ_xλ, s_l_αΔ_aff, s_u_αΔ_aff,
                                           x_m_l_αΔ_aff, u_m_x_αΔ_aff, rxs_l, rxs_u, max_iter, ϵ,
                                           start_time, max_time, c_catch, c_pdd, display)

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
        pt, pri_obj, res = post_scale(d1, d2, d3, pt, res, FloatData_T0, IntData,
                                      Qx, ATλ, Ax, cTx, pri_obj, dual_obj, xTQx_2)
    end

    elapsed_time = time() - start_time

    stats = GenericExecutionStats(status, QM, solution = pt.x[1:QM.meta.nvar],
                                  objective = pri_obj,
                                  dual_feas = res.rcNorm,
                                  primal_feas = res.rbNorm,
                                  multipliers = pt.λ,
                                  multipliers_L = pt.s_l,
                                  multipliers_U = pt.s_u,
                                  iter = k,
                                  elapsed_time=elapsed_time)
    return stats
end

end
