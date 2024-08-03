# Formulation K2: (if regul==:classic, adds additional regularization parmeters -ρ (top left) and δ (bottom right))
# [-Q - D     A' ] [x] = rhs
# [ A         0  ] [y]
export K2CUDSSParams

"""
Type to use the K2 formulation with an LDLᵀ factorization of cuDSS.
The package [`CUDSS.jl`](https://github.com/exanauts/CUDSS.jl) is used by default.
The outer constructor is

    sp = K2CUDSSParams(uplo, ρ, δ)
"""
struct K2CUDSSParams{T<:Real} <: SolverParams
    uplo :: Symbol # mandatory, tells RipQP which triangle of the augmented system to store
    ρ    :: T # dual regularization
    δ    :: T # primal regularization
end

mutable struct PreallocatedDataK2CUDSS{T<:Real, S} <: RipQP.PreallocatedDataAugmented{T, S}
    D      :: S # temporary top-left diagonal of the K2 system
    ρ      :: T # dual regularization
    δ      :: T # primal regularization
    K      :: CuSparseMatrixCSR{T,Cint} # K2 matrix
    K_fact :: CudssSolver{T} # factorized K2
end

function RipQP.PreallocatedData(sp :: K2CUDSSParams, fd :: RipQP.QM_FloatData{T},
                                id :: RipQP.QM_IntData, itd :: RipQP.IterData{T},
                                pt :: RipQP.Point{T},
                                iconf :: InputConfig{Tconf}) where {T<:Real, Tconf<:Real}

    ρ, δ = T(sp.ρ), T(sp.δ)
    K = CUDA.spzeros(T, id.ncon+id.nvar, id.ncon + id.nvar)
    K[1:id.nvar, 1:id.nvar] = .-fd.Q .- ρ .* Diagonal(ones(T, id.nvar))
    # A = Aᵀ of the input QuadraticModel since we use the upper triangle:
    K[1:id.nvar, id.nvar+1:end] = fd.A 
    K[diagind(K)[id.nvar+1:end]] .= δ

    K_fact = ldlt(Symmetric(K, sp.data))
    K_fact.__factorized = true

    return PreallocatedDataK2CUDSS(CUDA.zeros(T, id.nvar),
                                    ρ,
                                    δ,
                                    K, #K
                                    K_fact #K_fact
                                    )
end

function RipQP.update_pad!(pad :: PreallocatedDataK2basic{T}, dda :: RipQP.DescentDirectionAllocs{T},
                           pt :: RipQP.Point{T}, itd :: RipQP.IterData{T},
                           fd :: RipQP.Abstract_QM_FloatData{T}, id :: RipQP.QM_IntData,
                           res :: RipQP.Residuals{T}, cnts :: RipQP.Counters) where {T<:Real}

    # update the diagonal of K2
    pad.D .= -pad.ρ
    pad.D[id.ilow] .-= pt.s_l ./ itd.x_m_lvar
    pad.D[id.iupp] .-= pt.s_u ./ itd.uvar_m_x
    pad.D .-= fd.Q[diagind(fd.Q)]
    pad.K[diagind(pad.K)[1:id.nvar]] = pad.D
    pad.K[diagind(pad.K)[id.nvar+1:end]] .= pad.δ

    # factorize K2
    ldlt!(pad.K_fact, Symmetric(pad.K, :U))

end

function RipQP.solver!(dd :: AbstractVector{T}, pad :: PreallocatedDataK2basic{T},
                       dda :: RipQP.DescentDirectionAllocsPC{T}, pt :: RipQP.Point{T},
                       itd :: RipQP.IterData{T}, fd :: RipQP.Abstract_QM_FloatData{T},
                       id :: RipQP.QM_IntData, res :: RipQP.Residuals{T},
                       cnts :: RipQP.Counters, step :: Symbol) where {T<:Real}

    ldiv!(pad.K_fact, dd)
    return 0
end
