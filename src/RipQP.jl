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
    ϵ = tolerances(ϵ_pdd, ϵ_rb, ϵ_rc, one(T), one(T), ϵ_μ, ϵ_Δx)

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
        FloatData32, ϵ32, ϵ, regu, itd, pad, pt,
            res, optimal, small_Δx, small_μ = init_params(T, FloatData_T0, IntData, ϵ)

    elseif mode == :mono
        # init regularization values
        regu, itd, ϵ, pad, pt, res, optimal,
            small_Δx, small_μ = init_params_mono(FloatData_T0, IntData, ϵ)
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
        @info log_row(Any[k, itd.pri_obj, itd.pdd, res.rbNorm, res.rcNorm, res.n_Δx, zero(T), zero(T), itd.μ])
    end

    if mode == :multi
        # iters Float 32
        pt, res, itd,  Δt,
            tired, optimal, k, regu,
            c_catch, c_pdd  = iter_mehrotraPC!(pt, itd, FloatData32, IntData, res,
                                               small_Δx, small_μ, Δt, tired, optimal,
                                               k, regu, pad, 30, ϵ32, start_time, max_time, c_catch, c_pdd, display)

        # conversions to Float64
        T = Float64
        pt, itd, res, regu, pad = convert_types!(T, pt, itd, res, regu, pad)
        optimal = itd.pdd < ϵ_pdd && res.rbNorm < ϵ.tol_rb && res.rcNorm < ϵ.tol_rc
        small_Δx, small_μ = res.n_Δx < ϵ_Δx, itd.μ < ϵ_μ
        regu.ρ /= 10
        regu.δ /= 10
    end

    # iters T0
    pt, res, itd, Δt,
        tired, optimal, k, regu,
        c_catch, c_pdd  = iter_mehrotraPC!(pt, itd, FloatData_T0, IntData, res,
                                           small_Δx, small_μ, Δt, tired, optimal,
                                           k, regu, pad, max_iter, ϵ,
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
        pt, pri_obj, res = post_scale(d1, d2, d3, pt, res, FloatData_T0, IntData, itd.Qx,
                                      itd.ATλ, itd.Ax, itd.cTx, itd.pri_obj, itd.dual_obj, itd.xTQx_2)
    end

    elapsed_time = time() - start_time

    stats = GenericExecutionStats(status, QM, solution = pt.x[1:QM.meta.nvar],
                                  objective = itd.pri_obj,
                                  dual_feas = res.rcNorm,
                                  primal_feas = res.rbNorm,
                                  multipliers = pt.λ,
                                  multipliers_L = pt.s_l,
                                  multipliers_U = pt.s_u,
                                  iter = k,
                                  elapsed_time = elapsed_time)
    return stats
end

end
