import Base: convert

mutable struct QM_FloatData{T<:Real}
    Qvals :: Vector{T}
    Avals :: Vector{T}
    b     :: Vector{T}
    c     :: Vector{T}
    c0    :: T
    lvar  :: Vector{T}
    uvar  :: Vector{T}
end

mutable struct QM_IntData
    Qrows  :: Vector{Int}
    Qcols  :: Vector{Int}
    Arows  :: Vector{Int}
    Acols  :: Vector{Int}
    ilow   :: Vector{Int}
    iupp   :: Vector{Int}
    irng   :: Vector{Int}
    n_rows :: Int
    n_cols :: Int
    n_low  :: Int
    n_upp  :: Int
end

mutable struct tolerances{T<:Real}
    pdd    :: T
    rb     :: T
    rc     :: T
    tol_rb :: T
    tol_rc :: T
    μ      :: T
    Δx     :: T
end

mutable struct point{T<:Real}
    x    :: Vector{T}
    λ    :: Vector{T}
    s_l  :: Vector{T}
    s_u  :: Vector{T}
end

convert(::Type{point{T}}, pt) where {T<:Real} = point(convert(Array{T}, pt.x), convert(Array{T}, pt.λ),
                                                      convert(Array{T}, pt.s_l), convert(Array{T}, pt.s_u))

mutable struct residuals{T<:Real}
    rb      :: Vector{T}
    rc      :: Vector{T}
    rbNorm  :: T
    rcNorm  :: T
    n_Δx    :: T
end

convert(::Type{residuals{T}}, res) where {T<:Real} = residuals(convert(Array{T}, res.rb), convert(Array{T}, res.rc),
                                                               convert(T, res.rbNorm), convert(T, res.rcNorm),
                                                               convert(T, res.n_Δx))

mutable struct regularization{T<:Real}
    ρ      :: T
    δ      :: T
    ρ_min  :: T
    δ_min  :: T
end

convert(::Type{regularization{T}}, regu) where {T<:Real} = regularization(T(regu.ρ), T(regu.δ),
                                                                          T(regu.ρ_min), T(regu.δ_min))

mutable struct iter_data{T<:Real}
    tmp_diag    :: Vector{T}
    diag_Q      :: Vector{T}
    J_augm      :: SparseMatrixCSC{T,Int}
    J_fact      :: LDLFactorizations.LDLFactorization{T,Int,Int,Int}
    diagind_J   :: Vector{Int}
    x_m_lvar    :: Vector{T}
    uvar_m_x    :: Vector{T}
    Qx          :: Vector{T}
    ATλ         :: Vector{T}
    Ax          :: Vector{T}
    xTQx_2      :: T
    cTx         :: T
    pri_obj     :: T
    dual_obj    :: T
    μ           :: T
    pdd         :: T
    l_pdd       :: Vector{T}
    mean_pdd    :: T
end

createldl(T, J_fact) = LDLFactorizations.LDLFactorization(J_fact.__analyzed, J_fact.__factorized, J_fact.__upper,
                                                          J_fact.n, J_fact.parent, J_fact.Lnz, J_fact.flag,
                                                          J_fact.P, J_fact.pinv, J_fact.Lp, J_fact.Cp,
                                                          J_fact.Ci, J_fact.Li, Array{T}(J_fact.Lx),
                                                          Array{T}(J_fact.d), Array{T}(J_fact.Y),
                                                          J_fact.pattern)

convert(::Type{iter_data{T}}, itd) where {T<:Real} = iter_data(convert(Array{T}, itd.tmp_diag),
                                                                convert(Array{T}, itd.diag_Q),
                                                                convert(SparseMatrixCSC{T,Int}, itd.J_augm),
                                                                createldl(T, itd.J_fact),
                                                                itd.diagind_J,
                                                                convert(Array{T}, itd.x_m_lvar),
                                                                convert(Array{T}, itd.uvar_m_x),
                                                                convert(Array{T}, itd.Qx),
                                                                convert(Array{T}, itd.ATλ),
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
    Δ_aff            :: Vector{T}
    Δ_cc             :: Vector{T}
    Δ                :: Vector{T}
    Δ_xλ             :: Vector{T}
    x_m_l_αΔ_aff     :: Vector{T}
    u_m_x_αΔ_aff     :: Vector{T}
    s_l_αΔ_aff       :: Vector{T}
    s_u_αΔ_aff       :: Vector{T}
    rxs_l            :: Vector{T}
    rxs_u            :: Vector{T}
end

convert(::Type{preallocated_data{T}}, pad) where {T<:Real} = preallocated_data(convert(Array{T}, pad.Δ_aff),
                                                                               convert(Array{T}, pad.Δ_cc),
                                                                               convert(Array{T}, pad.Δ),
                                                                               convert(Array{T}, pad.Δ_xλ),
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

mutable struct safety_compt
    c_catch  :: Int
    c_pdd    :: Int
end
