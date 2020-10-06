mutable struct QM_FloatData{T}
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

mutable struct tolerances{T}
    pdd  :: T
    rb   :: T
    rc   :: T
    μ    :: T
    Δx   :: T
end

mutable struct point
    x    :: Vector
    λ    :: Vector
    s_l  :: Vector
    s_u  :: Vector
end

mutable struct residuals
    rb   :: Vector
    rc   :: Vector
    rbNorm
    rcNorm
end

mutable struct regularization
    ρ
    δ
    ρ_min
    δ_min
end

  
