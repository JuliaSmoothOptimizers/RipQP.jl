import Base: convert

# problem: min 1/2 x'Qx + c'x + c0     s.t.  Ax = b,  lvar ≤ x ≤ uvar
mutable struct QM_FloatData{T<:Real}
    Q     :: SparseMatrixCSC{T,Int}
    AT    :: SparseMatrixCSC{T,Int}
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

mutable struct regularization{T<:Real}
    ρ        :: T       # curent top-left regularization parameter
    δ        :: T       # cureent bottom-right regularization parameter
    ρ_min    :: T       # ρ minimum value 
    δ_min    :: T       # δ minimum value 
    regul    :: Symbol  # regularization mode (:classic, :dynamic, or :none)
end

convert(::Type{regularization{T}}, regu::regularization{T0}) where {T<:Real, T0<:Real} = 
    regularization(T(regu.ρ), T(regu.δ),
                   T(regu.ρ_min), T(regu.δ_min),
                   regu.regul)

mutable struct iter_data{T<:Real}
    tmp_diag    :: Vector{T}                                        # temporary top-left diagonal
    diag_Q      :: SparseVector{T,Int}                              # Q diagonal
    J_augm      :: SparseMatrixCSC{T,Int}                           # augmented matrix 
    J_fact      :: LDLFactorizations.LDLFactorization{T,Int,Int,Int}# factorized matrix
    diagind_J   :: Vector{Int}                                      # diagonal indices of J
    regu        :: regularization{T}
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
end

createldl(T, J_fact) = LDLFactorizations.LDLFactorization(J_fact.__analyzed, J_fact.__factorized, J_fact.__upper,
                                                          J_fact.n, J_fact.parent, J_fact.Lnz, J_fact.flag,
                                                          J_fact.P, J_fact.pinv, J_fact.Lp, J_fact.Cp,
                                                          J_fact.Ci, J_fact.Li, Array{T}(J_fact.Lx),
                                                          Array{T}(J_fact.d), Array{T}(J_fact.Y),
                                                          J_fact.pattern)

convert(::Type{iter_data{T}}, itd) where {T<:Real} = iter_data(convert(Array{T}, itd.tmp_diag),
                                                                convert(SparseVector{T,Int}, itd.diag_Q),
                                                                convert(SparseMatrixCSC{T,Int}, itd.J_augm),
                                                                createldl(T, itd.J_fact),
                                                                itd.diagind_J,
                                                                convert(regularization{T}, itd.regu),
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
                                                                convert(T, itd.mean_pdd)
                                                                )

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

mutable struct stop_crit
    optimal   :: Bool
    small_Δx  :: Bool
    small_μ   :: Bool
    tired     :: Bool 
end

mutable struct counters
    c_catch  :: Int # safety try:cath
    c_pdd    :: Int # maximum number of δ_min reductions when pdd does not change
    k        :: Int # iter count
    km       :: Int # iter relative to precision: if k+=1 and T==Float128, km +=16  (km+=4 if T==Float64 and km+=1 if T==Float32)
    K        :: Int # maximum corrector steps
end
