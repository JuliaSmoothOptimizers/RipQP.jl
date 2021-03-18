import Base: convert

export InputConfig, InputTol, SolverParams, PreallocatedData

# problem: min 1/2 x'Qx + c'x + c0     s.t.  Ax = b,  lvar ≤ x ≤ uvar
abstract type Abstract_QM_FloatData{T<:Real} end

mutable struct QM_FloatData{T<:Real} <: Abstract_QM_FloatData{T}
    Q     :: SparseMatrixCSC{T,Int}
    AT    :: SparseMatrixCSC{T,Int} # using AT is easier to form systems
    b     :: Vector{T}
    c     :: Vector{T}
    c0    :: T
    lvar  :: Vector{T}
    uvar  :: Vector{T}
end

mutable struct QM_IntData
    ilow   :: Vector{Int}
    iupp   :: Vector{Int}
    irng   :: Vector{Int}
    ifree  :: Vector{Int}
    n_rows :: Int
    n_cols :: Int
    n_low  :: Int
    n_upp  :: Int
end

"""
Abstract type for tuning the parameters of the different solvers. 
Each solver has its own `SolverParams` type.

The `SolverParams` currently implemented within RipQP are:

- [`RipQP.K2LDLParams`](@ref)
- [`RipQP.K2_5LDLParams`](@ref)

"""
abstract type SolverParams end 

"""
Type to specify the configuration used by RipQP.

- `mode :: Symbol`: should be `:mono` to use the mono-precision mode, or `:multi` to use
    the multi-precision mode (start in single precision and gradually transitions
    to `T0`)
- `scaling :: Bool`: activate/deactivate scaling of A and Q in `QM0`
- `normalize_rtol :: Bool = true` : if `true`, the primal and dual tolerance for the stopping criteria 
    are normalized by the initial primal and dual residuals
- `kc :: Int`: number of centrality corrections (set to `-1` for automatic computation)
- `refinement :: Symbol` : should be `:zoom` to use the zoom procedure, `:multizoom` to use the zoom procedure 
    with multi-precision (then `mode` should be `:multi`), `ref` to use the QP refinement procedure, `multiref` 
    to use the QP refinement procedure with multi_precision (then `mode` should be `:multi`), or `none` to avoid 
    refinements
- `sp :: SolverParams` : choose a solver to solve linear systems that occurs at each iteration and during the 
    initialization, see [`RipQP.SolverParams`](@ref)
- `solve_method :: Symbol` : used to solve the system at each iteration

The constructor

    iconf = InputConfig(; mode :: Symbol = :mono, scaling :: Bool = true, normalize_rtol :: Bool = true, kc :: I = 0, 
                        refinement :: Symbol = :none, max_ref :: I = 0, sp :: SolverParams = K2LDLParams(),
                        solve_method :: Symbol = :PC) where {I<:Integer}

returns a `InputConfig` struct that shall be used to solve the input `QuadraticModel` with RipQP.
"""
struct InputConfig{I<:Integer}
    mode                :: Symbol
    scaling             :: Bool 
    normalize_rtol      :: Bool # normalize the primal and dual tolerance to the initial starting primal and dual residuals
    kc                  :: I # multiple centrality corrections, -1 = automatic computation

    # QP refinement 
    refinement          :: Symbol 
    max_ref             :: I # maximum number of refinements

    # Functions to choose formulations
    sp                  :: SolverParams
    solve_method        :: Symbol
end

function InputConfig(; mode :: Symbol = :mono, scaling :: Bool = true, normalize_rtol :: Bool = true, 
                      kc :: I = 0, refinement :: Symbol = :none, max_ref :: I = 0, sp :: SolverParams = K2LDLParams(),
                      solve_method :: Symbol = :PC) where {I<:Integer}

    mode == :mono || mode == :multi || error("mode should be :mono or :multi")
    refinement == :zoom || refinement == :multizoom || refinement == :ref || refinement == :multiref || 
        refinement == :none || error("not a valid refinement parameter")
    solve_method == :IPF && kc != 0 && error("IPF method should not be used with centrality corrections") 

    return InputConfig{I}(mode, scaling, normalize_rtol, kc, refinement, max_ref, sp, solve_method)
end

"""
Type to specify the tolerances used by RipQP.

- `max_iter :: Int`: maximum number of iterations
- `ϵ_pdd`: relative primal-dual difference tolerance
- `ϵ_rb`: primal tolerance
- `ϵ_rc`: dual tolerance
- `max_iter32`, `ϵ_pdd32`, `ϵ_rb32`, `ϵ_rc32`: same as `max_iter`, `ϵ_pdd`, `ϵ_rb` and
    `ϵ_rc`, but used for switching from single precision to double precision. They are
    only usefull when `mode=:multi`
- `max_iter64`, `ϵ_pdd64`, `ϵ_rb64`, `ϵ_rc64`: same as `max_iter`, `ϵ_pdd`, `ϵ_rb` and
    `ϵ_rc`, but used for switching from double precision to quadruple precision. They
    are only usefull when `mode=:multi` and `T0=Float128`
- `ϵ_rbz` : primal transition tolerance for the zoom procedure, (used only if `refinement=:zoom`)
- `ϵ_Δx`: step tolerance for the current point estimate (note: this criterion
    is currently disabled)
- `ϵ_μ`: duality measure tolerance (note: this criterion is currently disabled)
- `max_time`: maximum time to solve the QP

The constructor

    itol = InputTol(;max_iter :: I = 200, max_iter32 :: I = 40, max_iter64 :: I = 180, 
                     ϵ_pdd :: T = 1e-8, ϵ_pdd32 :: T = 1e-2, ϵ_pdd64 :: T = 1e-4, 
                     ϵ_rb :: T = 1e-6, ϵ_rb32 :: T = 1e-4, ϵ_rb64 :: T = 1e-5, ϵ_rbz :: T = 1e-3,
                     ϵ_rc :: T = 1e-6, ϵ_rc32 :: T = 1e-4, ϵ_rc64 :: T = 1e-5,
                     ϵ_Δx :: T = 1e-16, ϵ_μ :: T = 1e-9) where {T<:Real, I<:Integer}

returns a `InputTol` struct that initializes the stopping criteria for RipQP. 
The 32 and 64 characters refer to the stopping criteria in `:multi` mode for the transitions from `Float32` to `Float64` 
and `Float64` to `Float128` (if the input `QuadraticModel` is in `Float128`) respectively.
"""
struct InputTol{T<:Real, I<:Integer}
    # maximum number of iterations
    max_iter        :: I
    max_iter32      :: I # only in multi mode
    max_iter64      :: I # only in multi mode with T0 = Float128

    # relative primal-dual gap tolerance
    ϵ_pdd           :: T
    ϵ_pdd32         :: T # only in multi mode 
    ϵ_pdd64         :: T # only in multi mode with T0 = Float128

    # primal residual
    ϵ_rb            :: T
    ϵ_rb32          :: T # only in multi mode 
    ϵ_rb64          :: T # only in multi mode with T0 = Float128
    ϵ_rbz           :: T # only when using zoom refinement

    # dual residual
    ϵ_rc            :: T 
    ϵ_rc32          :: T # only in multi mode 
    ϵ_rc64          :: T # only in multi mode with T0 = Float128

    # unused residuals (for now)
    ϵ_μ             :: T
    ϵ_Δx            :: T

    # maximum time for resolution
    max_time        :: T
end

function InputTol(;max_iter :: I = 200, max_iter32 :: I = 40, max_iter64 :: I = 180, 
                   ϵ_pdd :: T = 1e-8, ϵ_pdd32 :: T = 1e-2, ϵ_pdd64 :: T = 1e-4, 
                   ϵ_rb :: T = 1e-6, ϵ_rb32 :: T = 1e-4, ϵ_rb64 :: T = 1e-5, ϵ_rbz :: T = 1e-5,
                   ϵ_rc :: T = 1e-6, ϵ_rc32 :: T = 1e-4, ϵ_rc64 :: T = 1e-5,
                   ϵ_Δx :: T = 1e-16, ϵ_μ :: T = 1e-9, max_time :: T = 1200.) where {T<:Real, I<:Integer}

    return InputTol{T, I}(max_iter, max_iter32, max_iter64, ϵ_pdd, ϵ_pdd32, ϵ_pdd64, ϵ_rb, ϵ_rb32, ϵ_rb64, ϵ_rbz,
                           ϵ_rc, ϵ_rc32, ϵ_rc64, ϵ_μ, ϵ_Δx, max_time)
end

mutable struct Tolerances{T<:Real}
    pdd              :: T  # primal-dual difference (relative)
    rb               :: T  # primal residuals tolerance
    rc               :: T  # dual residuals tolerance
    tol_rb           :: T  # ϵ_rb * (1 + ||r_b0||)
    tol_rc           :: T  # ϵ_rc * (1 + ||r_c0||)
    μ                :: T  # duality measure
    Δx               :: T  
    normalize_rtol   :: Bool # true if normalize_rtol=true, then tol_rb, tol_rc = ϵ_rb, ϵ_rc
end

mutable struct Point{T<:Real}
    x    :: Vector{T}
    y    :: Vector{T}
    s_l  :: Vector{T}
    s_u  :: Vector{T}
end

convert(::Type{Point{T}}, pt) where {T<:Real} = Point(convert(Array{T}, pt.x), convert(Array{T}, pt.y),
                                                      convert(Array{T}, pt.s_l), convert(Array{T}, pt.s_u))

mutable struct Residuals{T<:Real}
    rb      :: Vector{T} # primal residuals
    rc      :: Vector{T} # dual residuals
    rbNorm  :: T       
    rcNorm  :: T
    n_Δx    :: T
end

convert(::Type{Residuals{T}}, res) where {T<:Real} = Residuals(convert(Array{T}, res.rb), convert(Array{T}, res.rc),
                                                               convert(T, res.rbNorm), convert(T, res.rcNorm),
                                                               convert(T, res.n_Δx))

# LDLFactorization conversion function
convertldl(T :: DataType, K_fact) = LDLFactorizations.LDLFactorization(K_fact.__analyzed, K_fact.__factorized, K_fact.__upper,
                                                                       K_fact.n, K_fact.parent, K_fact.Lnz, K_fact.flag,
                                                                       K_fact.P, K_fact.pinv, K_fact.Lp, K_fact.Cp,
                                                                       K_fact.Ci, K_fact.Li, convert(Array{T}, K_fact.Lx),
                                                                       convert(Array{T}, K_fact.d), convert(Array{T}, K_fact.Y),
                                                                       K_fact.pattern, T(K_fact.r1), T(K_fact.r2),
                                                                       T(K_fact.tol), K_fact.n_d)

mutable struct Regularization{T<:Real}
    ρ        :: T       # curent top-left regularization parameter
    δ        :: T       # cureent bottom-right regularization parameter
    ρ_min    :: T       # ρ minimum value 
    δ_min    :: T       # δ minimum value 
    regul    :: Symbol  # regularization mode (:classic, :dynamic, or :none)
end
                                                            
convert(::Type{Regularization{T}}, regu::Regularization{T0}) where {T<:Real, T0<:Real} = 
    Regularization(T(regu.ρ), T(regu.δ), T(regu.ρ_min), T(regu.δ_min), regu.regul)

mutable struct IterData{T<:Real} 
    Δxy         :: Vector{T}                                        # Newton step
    Δs_l        :: Vector{T} 
    Δs_u        :: Vector{T}
    x_m_lvar    :: Vector{T}                                        # x - lvar
    uvar_m_x    :: Vector{T}                                        # uvar - x
    Qx          :: Vector{T}                                        
    ATy         :: Vector{T}
    Ax          :: Vector{T}
    xTQx_2      :: T
    cTx         :: T
    pri_obj     :: T                                                
    dual_obj    :: T
    μ           :: T                                                # duality measure
    pdd         :: T                                                # primal dual difference (relative)
    l_pdd       :: Vector{T}                                        # list of the 5 last pdd
    mean_pdd    :: T                                                # mean of the 5 last pdd
    qp          :: Bool # true if qp false if lp
end

convert(::Type{IterData{T}}, itd :: IterData{T0}) where {T<:Real, T0<:Real} = 
    IterData(convert(Array{T}, itd.Δxy),
             convert(Array{T}, itd.Δs_l),
             convert(Array{T}, itd.Δs_u),
             convert(Array{T}, itd.x_m_lvar),
             convert(Array{T}, itd.uvar_m_x),
             convert(Array{T}, itd.Qx),
             convert(Array{T}, itd.ATy),
             convert(Array{T}, itd.Ax),
             convert(T, itd.xTQx_2),
             convert(T, itd.cTx),
             convert(T, itd.pri_obj),
             convert(T, itd.dual_obj),
             convert(T, itd.μ),
             convert(T, itd.pdd),
             convert(Array{T}, itd.l_pdd),
             convert(T, itd.mean_pdd),
             itd.qp
             )

abstract type PreallocatedData{T<:Real} end

mutable struct StopCrit{T}
    optimal     :: Bool
    small_Δx    :: Bool
    small_μ     :: Bool
    tired       :: Bool 
    max_iter    :: Int
    max_time    :: T
    start_time  :: T
    Δt          :: T
end

mutable struct Counters
    c_catch  :: Int # safety try:cath
    c_pdd    :: Int # maximum number of δ_min reductions when pdd does not change
    k        :: Int # iter count
    km       :: Int # iter relative to precision: if k+=1 and T==Float128, km +=16  (km+=4 if T==Float64 and km+=1 if T==Float32)
    kc       :: Int # maximum corrector steps
    max_ref  :: Int # maximum number of refinements
    c_ref    :: Int # current number of refinements
end
