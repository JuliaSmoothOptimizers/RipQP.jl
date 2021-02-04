module RipQP

using LinearAlgebra, Quadmath, SparseArrays, Statistics

using LDLFactorizations, QuadraticModels, SolverTools

export ripqp, iter_data, iter_data_K2, create_K2_iterdata, solve_K2!

include("types_definition.jl")
include("iterations/iterations.jl")
include("data_initialization.jl")
include("starting_points.jl")
include("scaling.jl")

"""
    ripqp(QM0; mode=:mono, regul=:classic, scaling=true, K=0,
          max_iter=200, ϵ_pdd=1e-8, ϵ_rb=1e-6, ϵ_rc=1e-6,
          max_iter32=40, ϵ_pdd32=1e-2, ϵ_rb32=1e-4, ϵ_rc32=1e-4,
          max_iter64=180, ϵ_pdd64=1e-4, ϵ_rb64=1e-5, ϵ_rc64=1e-5,
          ϵ_Δx=1e-16, ϵ_μ=1e-9, max_time=1200., display=true)

Minimize a convex quadratic problem. Algorithm stops when the criteria in pdd, rb, and rc are valid.
Returns a `GenericExecutionStats` containing information about the solved problem.

- `QM0::QuadraticModel{T0}`: problem to solve
- `mode::Symbol`: should be `:mono` to use the mono-precision mode, or `:multi` to use
    the multi-precision mode (start in single precision and gradually transitions
    to `T0`)
- `regul::Symbol`: if `:classic`, then the regularization is performed prior the factorization,
    if `:dynamic`, then the regularization is performed during the factorization, and if `:none`,
    no regularization is used
- `scaling::Bool`: activate/deactivate scaling of A and Q in `QM0`
- `K::Int`: number of centrality corrections (set to `-1` for automatic computation)
- `max_iter::Int`: maximum number of iterations
- `ϵ_pdd`: primal-dual difference tolerance
- `ϵ_rb`: primal tolerance
- `ϵ_rc`: dual tolerance
- `max_iter32`, `ϵ_pdd32`, `ϵ_rb32`, `ϵ_rc32`: same as `max_iter`, `ϵ_pdd`, `ϵ_rb` and
    `ϵ_rc`, but used for switching from single precision to double precision. They are
    only usefull when `mode=:multi`
- `max_iter64`, `ϵ_pdd64`, `ϵ_rb64`, `ϵ_rc64`: same as `max_iter`, `ϵ_pdd`, `ϵ_rb` and
    `ϵ_rc`, but used for switching from double precision to quadruple precision. They
    are only usefull when `mode=:multi` and `T0=Float128`
- `ϵ_Δx`: step tolerance for the current point estimate (note: this criterion
    is currently disabled)
- `ϵ_μ`: duality measure tolerance (note: this criterion is currently disabled)
- `max_time`: maximum time to solve `QM0`, in seconds
- `display::Bool`: activate/deactivate iteration data display
- `create_itd :: Function`: used to create the iter_data type used for the iterations (including the system 
    to solve)
- `solve! :: Function` : used to solve the system at each iteration
"""
function ripqp(QM :: QuadraticModel; mode :: Symbol = :mono, regul :: Symbol = :classic, scaling :: Bool = true,
               K :: Int = 0,
               max_iter :: Int = 200, ϵ_pdd :: Real = 1e-8, ϵ_rb :: Real = 1e-6, ϵ_rc :: Real = 1e-6,
               max_iter32 :: Int = 40, ϵ_pdd32 :: Real = 1e-2, ϵ_rb32 :: Real = 1e-4, ϵ_rc32 :: Real = 1e-4,
               max_iter64 :: Int = 180, ϵ_pdd64 :: Real = 1e-4, ϵ_rb64 :: Real = 1e-5, ϵ_rc64 :: Real = 1e-5, # params for the itermediate ϵ in :multi mode
               ϵ_Δx :: Real = 1e-16, ϵ_μ :: Real = 1e-9, max_time :: Real = 1200., display :: Bool = true,
               create_itd :: Function = create_K2_iterdata, solve! :: Function = solve_K2!)

    if mode ∉ [:mono, :multi]
        error("mode should be :mono or :multi")
    end
    if regul ∉ [:classic, :dynamic, :none]
        error("regul should be :classic or :dynamic or :none")
    end
    
    start_time = time()
    elapsed_time = 0.0
    sc = stop_crit(false, false, false, false, max_iter, max_time, start_time, 0.)    
    
    nvar_init = QM.meta.nvar
    SlackModel!(QM) # add slack variables to the problem if QM.meta.lcon != QM.meta.ucon

    fd_T0, id, T = get_QM_data(QM)
    T0 = T # T0 is the data type, in mode :multi T will gradually increase to T0
    ϵ = tolerances(T(ϵ_pdd), T(ϵ_rb), T(ϵ_rc), one(T), one(T), T(ϵ_μ), T(ϵ_Δx))

    if scaling
        fd_T0, d1, d2, d3 = scaling_Ruiz!(fd_T0, id, T(1.0e-3))
    end

    # initialization
    if mode == :multi
        T = Float32
        ϵ32 = tolerances(T(ϵ_pdd32), T(ϵ_rb32), T(ϵ_rc32), one(T), one(T), T(ϵ_μ), T(ϵ_Δx))
        fd32 = convert_FloatData(T, fd_T0)
        itd, ϵ32, pad, pt, res, sc = init_params(fd32, id, ϵ32, sc, regul, mode, create_itd)
        ϵ.tol_rb, ϵ.tol_rc = ϵ.rb*(one(T0) + T0(res.rbNorm)), ϵ.rc*(one(T0) + T0(res.rcNorm))
    elseif mode == :mono
        itd, ϵ, pad, pt, res, sc = init_params(fd_T0, id, ϵ, sc, regul, mode, create_itd)
    end

    Δt = time() - start_time
    sc.tired = Δt > max_time
    
    cnts = counters(zero(Int), zero(Int), 0, 0, 
                    K==-1 ? nb_corrector_steps(itd.J_augm.colptr, id.n_rows, id.n_cols, T) : K)
    
    # display
    if display == true
        @info log_header([:k, :pri_obj, :pdd, :rbNorm, :rcNorm, :n_Δx, :α_pri, :α_du, :μ, :ρ, :δ],
        [Int, T, T, T, T, T, T, T, T, T, T, T],
        hdr_override=Dict(:k => "iter", :pri_obj => "obj", :pdd => "rgap",
        :rbNorm => "‖rb‖", :rcNorm => "‖rc‖",
        :n_Δx => "‖Δx‖"))
        @info log_row(Any[cnts.k, itd.pri_obj, itd.pdd, res.rbNorm, res.rcNorm, res.n_Δx, zero(T), zero(T), itd.μ, itd.regu.ρ, itd.regu.δ])
    end

    if mode == :multi
        # iters Float 32
        sc.max_iter = max_iter32
        iter_mehrotraPC!(pt, itd, fd32, id, res, sc, pad, ϵ32, cnts, solve!, T0, display)
        # convert to Float64
        T = Float64
        pt, itd, res, pad = convert_types!(T, pt, itd, res, pad, T0)
        sc.optimal = itd.pdd < ϵ_pdd && res.rbNorm < ϵ.tol_rb && res.rcNorm < ϵ.tol_rc
        sc.small_Δx, sc.small_μ = res.n_Δx < ϵ.Δx, itd.μ < ϵ.μ

        if T0 == Float128 # iters Float64 if T0 == Float128
            sc.max_iter = max_iter64
            fd64 = convert_fd(T, fd_T0)
            ϵ64 = tolerances(T(ϵ_pdd64), T(ϵ_rb64), T(ϵ_rc64), one(T), one(T), T(ϵ_μ), T(ϵ_Δx))
            ϵ64.tol_rb, ϵ64.tol_rc = ϵ64.rb*(one(T) + res.rbNorm), ϵ64.rc*(one(T) + res.rcNorm)
            iter_mehrotraPC!(pt, itd, fd64, id, res, sc, pad, ϵ64, cnts, solve!, T0, display)
            T = Float128
            pt, itd, res, regu, pad = convert_types!(T, pt, itd, res, pad, T0)
            sc.optimal = itd.pdd < ϵ_pdd && res.rbNorm < ϵ.tol_rb && res.rcNorm < ϵ.tol_rc
            sc.small_Δx, sc.small_μ = res.n_Δx < ϵ.Δx, itd.μ < ϵ.μ
        end
        sc.max_iter = max_iter
    end

    # iters T0
    iter_mehrotraPC!(pt, itd, fd_T0, id, res, sc, pad, ϵ, cnts, solve!, T0, display)

    # output status                                                    
    if cnts.k>= max_iter
        status = :max_iter
    elseif sc.tired
        status = :max_time
    elseif sc.optimal
        status = :acceptable
    else
        status = :unknown
    end

    if scaling
        pt, pri_obj, res = post_scale(d1, d2, d3, pt, res, fd_T0, id, itd.Qx, itd.ATy,
                                      itd.Ax, itd.cTx, itd.pri_obj, itd.dual_obj, itd.xTQx_2)
    end

    elapsed_time = time() - start_time

    stats = GenericExecutionStats(status, QM, solution = pt.x[1:nvar_init],
                                  objective = itd.pri_obj,
                                  dual_feas = res.rcNorm,
                                  primal_feas = res.rbNorm,
                                  multipliers = pt.y,
                                  multipliers_L = pt.s_l,
                                  multipliers_U = pt.s_u,
                                  iter = cnts.km,
                                  elapsed_time = elapsed_time,
                                  solver_specific = Dict(:absolute_iter_cnt=>cnts.k))
    return stats
end

end
