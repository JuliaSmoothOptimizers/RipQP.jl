import Base: convert

export input_config, input_tol

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
    n_rows :: Int
    n_cols :: Int
    n_low  :: Int
    n_upp  :: Int
end

"""
Type to specify the configuration used by RipQP.

- `mode :: Symbol`: should be `:mono` to use the mono-precision mode, or `:multi` to use
    the multi-precision mode (start in single precision and gradually transitions
    to `T0`)
- `regul :: Symbol`: if `:classic`, then the regularization is performed prior the factorization,
    if `:dynamic`, then the regularization is performed during the factorization, and if `:none`,
    no regularization is used
- `scaling :: Bool`: activate/deactivate scaling of A and Q in `QM0`
- `normalize_rtol :: Bool = true` : if `true`, the primal and dual tolerance for the stopping criteria 
    are normalized by the initial primal and dual residuals
- `K :: Int`: number of centrality corrections (set to `-1` for automatic computation)
- `refinement :: Symbol` : should be `:zoom` to use the zoom procedure, `:multizoom` to use the zoom procedure 
    with multi-precision (then `mode` should be `:multi`), `ref` to use the QP refinement procedure, `multiref` 
    to use the QP refinement procedure with multi_precision (then `mode` should be `:multi`), or `none` to avoid 
    refinements
- `create_iterdata :: Function`: used to create the iter_data type used for the iterations (including the system 
    to solve)
- `solve! :: Function` : used to solve the system at each iteration

The constructor

    iconf = input_config(; mode :: Symbol = :mono, regul :: Symbol = :classic, 
                         scaling :: Bool = true, normalize_rtol :: Bool = true, 
                         K :: I = 0, refinement :: Symbol = :none, max_ref :: I = 0, 
                         create_iterdata :: Function = create_iterdata_K2, 
                         solve! :: Function = solve_K2!) where {I<:Integer}

returns a `input_config` struct that shall be used to solve the input `QuadraticModel` with RipQP.
"""
struct input_config{I<:Integer}
    mode                :: Symbol
    regul               :: Symbol
    scaling             :: Bool 
    normalize_rtol      :: Bool # normalize the primal and dual tolerance to the initial starting primal and dual residuals
    K                   :: I # multiple centrality corrections, -1 = automatic computation

    # QP refinement 
    refinement          :: Symbol 
    max_ref             :: I # maximum number of refinements

    # Functions to choose formulations
    create_iterdata     :: Function 
    solve!              :: Function
end

function input_config(; mode :: Symbol = :mono, regul :: Symbol = :classic, scaling :: Bool = true, normalize_rtol :: Bool = true, 
                      K :: I = 0, refinement :: Symbol = :none, max_ref :: I = 0, 
                      create_iterdata :: Function = create_iterdata_K2, solve! :: Function = solve_K2!) where {I<:Integer}

    mode == :mono || mode == :multi || error("mode should be :mono or :multi")
    regul == :classic || regul == :dynamic || regul == :none || error("regul should be :classic or :dynamic or :none")
    refinement == :zoom || refinement == :multizoom || refinement == :ref || refinement == :multiref || 
        refinement == :none || error("not a valid refinement parameter")

    return input_config{I}(mode, regul, scaling, normalize_rtol, K, refinement, max_ref, create_iterdata, solve!)
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

    itol = input_tol(;max_iter :: I = 200, max_iter32 :: I = 40, max_iter64 :: I = 180, 
                     ϵ_pdd :: T = 1e-8, ϵ_pdd32 :: T = 1e-2, ϵ_pdd64 :: T = 1e-4, 
                     ϵ_rb :: T = 1e-6, ϵ_rb32 :: T = 1e-4, ϵ_rb64 :: T = 1e-5, ϵ_rbz :: T = 1e-3,
                     ϵ_rc :: T = 1e-6, ϵ_rc32 :: T = 1e-4, ϵ_rc64 :: T = 1e-5,
                     ϵ_Δx :: T = 1e-16, ϵ_μ :: T = 1e-9) where {T<:Real, I<:Integer}

returns a `input_tol` struct that initializes the stopping criteria for RipQP. 
The 32 and 64 characters refer to the stopping criteria in `:multi` mode for the transitions from `Float32` to `Float64` 
and `Float64` to `Float128` (if the input `QuadraticModel` is in `Float128`) respectively.
"""
struct input_tol{T<:Real, I<:Integer}
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

function input_tol(;max_iter :: I = 200, max_iter32 :: I = 40, max_iter64 :: I = 180, 
                   ϵ_pdd :: T = 1e-8, ϵ_pdd32 :: T = 1e-2, ϵ_pdd64 :: T = 1e-4, 
                   ϵ_rb :: T = 1e-6, ϵ_rb32 :: T = 1e-4, ϵ_rb64 :: T = 1e-5, ϵ_rbz :: T = 1e-5,
                   ϵ_rc :: T = 1e-6, ϵ_rc32 :: T = 1e-4, ϵ_rc64 :: T = 1e-5,
                   ϵ_Δx :: T = 1e-16, ϵ_μ :: T = 1e-9, max_time :: T = 1200.) where {T<:Real, I<:Integer}

    return input_tol{T, I}(max_iter, max_iter32, max_iter64, ϵ_pdd, ϵ_pdd32, ϵ_pdd64, ϵ_rb, ϵ_rb32, ϵ_rb64, ϵ_rbz,
                           ϵ_rc, ϵ_rc32, ϵ_rc64, ϵ_μ, ϵ_Δx, max_time)
end

mutable struct tolerances{T<:Real}
    pdd              :: T  # primal-dual difference (relative)
    rb               :: T  # primal residuals tolerance
    rc               :: T  # dual residuals tolerance
    tol_rb           :: T  # ϵ_rb * (1 + ||r_b0||)
    tol_rc           :: T  # ϵ_rc * (1 + ||r_c0||)
    μ                :: T  # duality measure
    Δx               :: T  
    normalize_rtol   :: Bool # true if normalize_rtol=true, then tol_rb, tol_rc = ϵ_rb, ϵ_rc
end

mutable struct point{T<:Real}
    x    :: Vector{T}
    y    :: Vector{T}
    s_l  :: Vector{T}
    s_u  :: Vector{T}
end

convert(::Type{point{T}}, pt) where {T<:Real} = point(convert(Array{T}, pt.x), convert(Array{T}, pt.y),
                                                      convert(Array{T}, pt.s_l), convert(Array{T}, pt.s_u))

mutable struct residuals{T<:Real}
    rb      :: Vector{T} # primal residuals
    rc      :: Vector{T} # dual residuals
    rbNorm  :: T       
    rcNorm  :: T
    n_Δx    :: T
end

convert(::Type{residuals{T}}, res) where {T<:Real} = residuals(convert(Array{T}, res.rb), convert(Array{T}, res.rc),
                                                               convert(T, res.rbNorm), convert(T, res.rcNorm),
                                                               convert(T, res.n_Δx))

abstract type iter_data{T<:Real} end

createldl(T, J_fact) = LDLFactorizations.LDLFactorization(J_fact.__analyzed, J_fact.__factorized, J_fact.__upper,
                                                          J_fact.n, J_fact.parent, J_fact.Lnz, J_fact.flag,
                                                          J_fact.P, J_fact.pinv, J_fact.Lp, J_fact.Cp,
                                                          J_fact.Ci, J_fact.Li, Array{T}(J_fact.Lx),
                                                          Array{T}(J_fact.d), Array{T}(J_fact.Y),
                                                          J_fact.pattern, T(J_fact.r1), T(J_fact.r2),
                                                          T(J_fact.tol), J_fact.n_d)

abstract type preallocated_data{T<:Real} end

mutable struct stop_crit{T}
    optimal     :: Bool
    small_Δx    :: Bool
    small_μ     :: Bool
    tired       :: Bool 
    max_iter    :: Int
    max_time    :: T
    start_time  :: T
    Δt          :: T
end

mutable struct counters
    c_catch  :: Int # safety try:cath
    c_pdd    :: Int # maximum number of δ_min reductions when pdd does not change
    k        :: Int # iter count
    km       :: Int # iter relative to precision: if k+=1 and T==Float128, km +=16  (km+=4 if T==Float64 and km+=1 if T==Float32)
    K        :: Int # maximum corrector steps
    max_ref  :: Int # maximum number of refinements
    c_ref    :: Int # current number of refinements
end
