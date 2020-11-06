module RipQP

using LinearAlgebra, Quadmath, SparseArrays, Statistics

using LDLFactorizations, NLPModels, QuadraticModels, SolverTools

export ripqp

include("types_definition.jl")
include("types_toolbox.jl")
include("starting_points.jl")
include("scaling.jl")
include("sparse_toolbox.jl")
include("iterations.jl")

function ripqp(QM0 :: AbstractNLPModel; mode :: Symbol = :mono,
               max_iter :: Int = 800, ϵ_pdd :: Real = 1e-8, ϵ_rb :: Real = 1e-6, ϵ_rc :: Real = 1e-6,
               max_iter32 :: Int = 40, max_iter64 :: Int = 600,
               ϵ_pdd32 :: Real = 1e-2, ϵ_rb32 :: Real = 1e-4, ϵ_rc32 :: Real = 1e-4,
               ϵ_pdd64 :: Real = 1e-4, ϵ_rb64 :: Real = 1e-5, ϵ_rc64 :: Real = 1e-5, # params for the itermediate ϵ in :multi mode
               ϵ_Δx :: Real = 1e-16, ϵ_μ :: Real = 1e-9, max_time :: Real = 1200.,
               scaling :: Bool = true, display :: Bool = true)

    if mode ∉ [:mono, :multi]
        error("mode should be :mono or :multi")
    end
    start_time = time()
    elapsed_time = 0.0
    QM = SlackModel(QM0)
    FloatData_T0, IntData, T = get_QM_data(QM)
    T0 = T # T0 is the data type, in mode :multi T will gradually increase to T0
    ϵ = tolerances(T(ϵ_pdd), T(ϵ_rb), T(ϵ_rc), one(T), one(T), T(ϵ_μ), T(ϵ_Δx))
    if scaling
        FloatData_T0, d1, d2, d3 = scaling_Ruiz!(FloatData_T0, IntData, T(1.0e-3))
    end
    # cNorm = norm(c)
    # bNorm = norm(b)
    # ANorm = norm(Avals)
    # QNorm = norm(Qvals)

    # initialization
    if mode == :multi
        T = Float32
        ϵ32 = tolerances(T(ϵ_pdd32), T(ϵ_rb32), T(ϵ_rc32), one(T), one(T), T(ϵ_μ), T(ϵ_Δx))
        FloatData32, ϵ32, ϵ, regu, itd, pad, pt,res, sc = init_params(FloatData_T0, IntData, ϵ32, ϵ)
    elseif mode == :mono
        regu, itd, ϵ, pad, pt, res, sc = init_params_mono(FloatData_T0, IntData, ϵ)
    end

    Δt = time() - start_time
    sc.tired = Δt > max_time
    k = 0
    safe = safety_compt(zero(Int), zero(Int)) # c_catch to avoid endless loop, c_pdd avoid too small δ_min

    # display
    if display == true
        @info log_header([:k, :pri_obj, :pdd, :rbNorm, :rcNorm, :n_Δx, :α_pri, :α_du, :μ, :ρ, :δ],
        [Int, T, T, T, T, T, T, T, T, T, T, T],
        hdr_override=Dict(:k => "iter", :pri_obj => "obj", :pdd => "rgap",
        :rbNorm => "‖rb‖", :rcNorm => "‖rc‖",
        :n_Δx => "‖Δx‖"))
        @info log_row(Any[k, itd.pri_obj, itd.pdd, res.rbNorm, res.rcNorm, res.n_Δx, zero(T), zero(T), itd.μ, regu.ρ, regu.δ])
    end

    if mode == :multi
        # iters Float 32
        pt, res, itd, Δt, sc, k, regu, safe  = iter_mehrotraPC!(pt, itd, FloatData32, IntData, res, sc, Δt, k, regu, pad,
                                                                max_iter32, ϵ32, start_time, max_time, safe, T0, display)
        # conversions to Float64
        T = Float64
        pt, itd, res, regu, pad = convert_types!(T, pt, itd, res, regu, pad, T0)
        sc.optimal = itd.pdd < ϵ_pdd && res.rbNorm < ϵ.tol_rb && res.rcNorm < ϵ.tol_rc
        sc.small_Δx, sc.small_μ = res.n_Δx < ϵ.Δx, itd.μ < ϵ.μ

        if T0 == Float128 # iters Float64 if T0 == Float128
            FloatData64 = convert_FloatData(T, FloatData_T0)
            ϵ64 = tolerances(T(ϵ_pdd64), T(ϵ_rb64), T(ϵ_rc64), one(T), one(T), T(ϵ_μ), T(ϵ_Δx))
            ϵ64.tol_rb, ϵ64.tol_rc = ϵ64.rb*(one(T) + res.rbNorm), ϵ64.rc*(one(T) + res.rcNorm)
            pt, res, itd,  Δt, sc, k, regu, safe  = iter_mehrotraPC!(pt, itd, FloatData64, IntData, res, sc, Δt, k, regu, pad,
                                                                     max_iter64, ϵ64, start_time, max_time, safe, T0, display)
            T = Float128
            pt, itd, res, regu, pad = convert_types!(T, pt, itd, res, regu, pad, T0)
            sc.optimal = itd.pdd < ϵ_pdd && res.rbNorm < ϵ.tol_rb && res.rcNorm < ϵ.tol_rc
            sc.small_Δx, sc.small_μ = res.n_Δx < ϵ.Δx, itd.μ < ϵ.μ
        end
    end

    # iters T0
    pt, res, itd, Δt, sc, k, regu, safe  = iter_mehrotraPC!(pt, itd, FloatData_T0, IntData, res, sc, Δt, k, regu, pad,
                                                            max_iter, ϵ, start_time, max_time, safe, T0, display)
    if k>= max_iter
        status = :max_iter
    elseif sc.tired
        status = :max_time
    elseif sc.optimal
        status = :acceptable
    else
        status = :unknown
    end

    if scaling
        pt, pri_obj, res = post_scale(d1, d2, d3, pt, res, FloatData_T0, IntData, itd.Qx, itd.ATλ,
                                      itd.Ax, itd.cTx, itd.pri_obj, itd.dual_obj, itd.xTQx_2)
    end

    elapsed_time = time() - start_time

    stats = GenericExecutionStats(status, QM, solution = pt.x[1:QM0.meta.nvar],
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
