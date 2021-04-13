struct IntDataInit{I<:Integer}
    nvar :: I
    ncon :: I
    ilow :: Vector{I}
    iupp :: Vector{I}
    irng :: Vector{I}
    ifix :: Vector{I}
    jlow :: Vector{I}
    jupp :: Vector{I}
    jrng :: Vector{I}
    jfix :: Vector{I}
end

function get_multipliers(s_l :: Vector{T}, s_u :: Vector{T}, y :: Vector{T}, idi :: IntDataInit{Int}) where {T<:Real}

    nlow, nupp, nrng = length(idi.ilow), length(idi.iupp), length(idi.irng)
    njlow, njupp, njrng = length(idi.jlow), length(idi.jupp), length(idi.jrng)
   
    multipliers_L = SparseVector(idi.nvar, [idi.ilow; idi.irng], s_l[1:nlow+nrng])
    multipliers_U = SparseVector(idi.nvar, [idi.iupp; idi.irng], s_u[1:nupp+nrng])
    multipliers = zeros(T, idi.ncon)
    multipliers[idi.jfix] .= @views y[idi.jfix] 
    multipliers[idi.jlow] .+= @views s_l[nlow+nrng+1: nlow+nrng+njlow]
    multipliers[idi.jupp] .-= @views s_u[nupp+nrng+1: nupp+nrng+njupp]
    multipliers[idi.jrng] .+= @views s_l[nlow+nrng+njlow+1: end] .- s_u[nlow+nrng+njupp+1: end]

    return multipliers, multipliers_L, multipliers_U
end
