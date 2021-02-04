import Base: convert

# problem: min 1/2 x'Qx + c'x + c0     s.t.  Ax = b,  lvar ≤ x ≤ uvar
mutable struct QM_FloatData{T<:Real}
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

mutable struct tolerances{T<:Real}
    pdd    :: T  # primal-dual difference (relative)
    rb     :: T  # primal residuals tolerance
    rc     :: T  # dual residuals tolerance
    tol_rb :: T  # ϵ_rb * (1 + ||r_b0||)
    tol_rc :: T  # ϵ_rc * (1 + ||r_c0||)
    μ      :: T  # duality measure
    Δx     :: T  
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
                                                          J_fact.pattern)

mutable struct preallocated_data{T<:Real}
    Δ_aff            :: Vector{T} # affine-step solution of the augmented system
    Δ_cc             :: Vector{T} # corrector-centering step solution of the augmented system
    Δ                :: Vector{T} # Δ_aff + Δ_cc
    Δ_xy             :: Vector{T} # temporary vector
    x_m_l_αΔ_aff     :: Vector{T} # x + α * Δ_aff - lvar
    u_m_x_αΔ_aff     :: Vector{T} # uvar - (x + α * Δ_aff)
    s_l_αΔ_aff       :: Vector{T} # s_l + α * Δ_aff
    s_u_αΔ_aff       :: Vector{T} # s_u + α * Δ_aff
    rxs_l            :: Vector{T} # - σ * μ * e + ΔX_aff * Δ_S_l_aff
    rxs_u            :: Vector{T} # σ * μ * e + ΔX_aff * Δ_S_u_aff
end

convert(::Type{preallocated_data{T}}, pad) where {T<:Real} = preallocated_data(convert(Array{T}, pad.Δ_aff),
                                                                               convert(Array{T}, pad.Δ_cc),
                                                                               convert(Array{T}, pad.Δ),
                                                                               convert(Array{T}, pad.Δ_xy),
                                                                               convert(Array{T}, pad.x_m_l_αΔ_aff),
                                                                               convert(Array{T}, pad.u_m_x_αΔ_aff),
                                                                               convert(Array{T}, pad.s_l_αΔ_aff),
                                                                               convert(Array{T}, pad.s_u_αΔ_aff),
                                                                               convert(Array{T}, pad.rxs_l),
                                                                               convert(Array{T}, pad.rxs_u)
                                                                               )

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
end
