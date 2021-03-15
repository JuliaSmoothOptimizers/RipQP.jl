# Formulation K2: (if regul==:classic, adds additional regularization parmeters -ρ (top left) and δ (bottom right))
# [-Q_X - D      sqrt(X1X2)A' ] [̃x] = rhs
# [ A sqrt(X1x2)         0    ] [y]
# where Q_X = sqrt(X1X2) Q sqrt(X1X2) and D = s_l X2 + s_u X1
# and  Δ x = sqrt(X1 X2) Δ ̃x
 
mutable struct preallocated_data_K2_5{T<:Real} <: preallocated_data{T} 
    diag_Q           :: SparseVector{T,Int} # Q diagonal
    K                :: SparseMatrixCSC{T,Int} # augmented matrix 
    K_fact           :: LDLFactorizations.LDLFactorization{T,Int,Int,Int} # factorized matrix
    fact_fail        :: Bool # true if factorization failed 
    diagind_K        :: Vector{Int} # diagonal indices of J
    Δxy_aff          :: Vector{T} # affine-step solution of the augmented system
    Δs_l_aff         :: Vector{T}
    Δs_u_aff         :: Vector{T}
    Δxy              :: Vector{T} # Newton step
    Δs_l             :: Vector{T} 
    Δs_u             :: Vector{T} 
    x_m_l_αΔ_aff     :: Vector{T} # x + α * Δxy_aff - lvar
    u_m_x_αΔ_aff     :: Vector{T} # uvar - (x + α * Δxy_aff)
    s_l_αΔ_aff       :: Vector{T} # s_l + α * Δs_l_aff
    s_u_αΔ_aff       :: Vector{T} # s_u + α * Δs_u_aff
    rxs_l            :: Vector{T} # - σ * μ * e + ΔX_aff * Δ_S_l_aff
    rxs_u            :: Vector{T} # σ * μ * e + ΔX_aff * Δ_S_u_aff
end

function preallocated_data_K2_5(fd :: QM_FloatData{T}, id :: QM_IntData, iconf :: input_config{Tconf}) where {T<:Real, Tconf<:Real} 
    # init regularization values
    if iconf.mode == :mono
        regu = regularization(T(sqrt(eps())*1e5), T(sqrt(eps())*1e5), 1e-5*sqrt(eps(T)), 1e0*sqrt(eps(T)), iconf.regul)
        D = -T(1.0e0)/2 .* ones(T, id.n_cols)
    else
        regu = regularization(T(sqrt(eps())*1e5), T(sqrt(eps())*1e5), T(sqrt(eps(T))*1e0), T(sqrt(eps(T))*1e0), iconf.regul)
        D = -T(1.0e-2) .* ones(T, id.n_cols)
    end
    diag_Q = get_diag_Q(fd.Q.colptr, fd.Q.rowval, fd.Q.nzval, id.n_cols)
    K = create_K2(id, D, fd.Q, fd.AT, diag_Q, regu)

    diagind_K = get_diag_sparseCSC(K.colptr, id.n_rows+id.n_cols)
    K_fact = ldl_analyze(Symmetric(K, :U))
    if regu.regul == :dynamic
        Amax = @views norm(K.nzval[diagind_K], Inf)
        K_fact.r1 = T(-eps(T)^(3/4))
        K_fact.r2 = T(sqrt(eps(T)))
        K_fact.tol = Amax*T(eps(T))
        K_fact.n_d = id.n_cols
        K_fact = ldl_factorize!(Symmetric(K, :U), K_fact)
    end
    K_fact = ldl_factorize!(Symmetric(K, :U), K_fact)
    K_fact.__factorized = true

    return preallocated_data_K2(D,
                                regu,
                                diag_Q, #diag_Q
                                K, #K
                                K_fact, #K_fact
                                false,
                                diagind_K, #diagind_K
                                zeros(T, id.n_cols+id.n_rows), # Δxy_aff
                                zeros(T, id.n_low), # Δs_l_aff
                                zeros(T, id.n_upp), # Δs_u_aff
                                zeros(T, id.n_cols+id.n_rows), # Δxy
                                zeros(T, id.n_low), # Δs_l
                                zeros(T, id.n_upp), # Δs_u
                                zeros(T, id.n_low), # x_m_l_αΔ_aff
                                zeros(T, id.n_upp), # u_m_x_αΔ_aff
                                zeros(T, id.n_low), # s_l_αΔ_aff
                                zeros(T, id.n_upp), # s_u_αΔ_aff
                                zeros(T, id.n_low), # rxs_l
                                zeros(T, id.n_upp)  # rxs_u
                                )
end
                
convert(::Type{<:preallocated_data{T}}, pad :: preallocated_data_K2_5{T0}) where {T<:Real, T0<:Real} = 
    preallocated_data_K2_5(convert(SparseVector{T,Int}, pad.diag_Q),
                         convert(SparseMatrixCSC{T,Int}, pad.K),
                         convertldl(T, pad.K_fact),
                         pad.fact_fail,
                         pad.diagind_K,
                         convert(Array{T}, pad.Δxy_aff),
                         convert(Array{T}, pad.Δs_l_aff),
                         convert(Array{T}, pad.Δs_u_aff),
                         convert(Array{T}, pad.Δxy),
                         convert(Array{T}, pad.Δs_l),
                         convert(Array{T}, pad.Δs_u),
                         convert(Array{T}, pad.x_m_l_αΔ_aff),
                         convert(Array{T}, pad.u_m_x_αΔ_aff),
                         convert(Array{T}, pad.s_l_αΔ_aff),
                         convert(Array{T}, pad.s_u_αΔ_aff),
                         convert(Array{T}, pad.rxs_l),
                         convert(Array{T}, pad.rxs_u)
                         )

# solver LDLFactorization
function solver!(pt :: point{T}, itd :: iter_data{T}, fd :: Abstract_QM_FloatData{T}, id :: QM_IntData, res :: residuals{T}, 
                 pad :: preallocated_data_K2_5{T}, cnts :: counters, T0 :: DataType, step :: Symbol) where {T<:Real} 

    if step == :init 
        LDLFactorizations.ldiv!(pad.K_fact, pad.Δxy)
    elseif step == :aff 
        out = factorize_K2_5!(pad.K, pad.K_fact, pad.D, pad.diag_Q, pad.diagind_K, pad.regu, 
                              pt.s_l, pt.s_u, itd.x_m_lvar, itd.uvar_m_x, id.ilow, id.iupp, 
                              id.n_rows, id.n_cols, cnts, itd.qp, T, T0)
        out == 1 && return out

        pad.Δxy_aff[1:id.n_cols] .*= pad.D
        LDLFactorizations.ldiv!(pad.K_fact, pad.Δxy_aff) 
        pad.Δxy_aff[1:id.n_cols] .*= pad.D
    else
        pad.Δxy[1:id.n_cols] .*= pad.D
        LDLFactorizations.ldiv!(pad.K_fact, pad.Δxy)
        pad.Δxy[1:id.n_cols] .*= pad.D

        if pad.regu.regul == :classic  # update ρ and δ values, check K diag magnitude 
            out = update_regu_diagJ_K2_5!(pad.regu, pad.D, itd.pdd, itd.l_pdd, itd.mean_pdd, cnts, T, T0) 
        end
        out == 1 && return out
    
        # restore J for next iteration
        pad.D .= one(T) 
        pad.D[id.ilow] ./= sqrt.(itd.x_m_lvar)
        pad.D[id.iupp] ./= sqrt.(itd.uvar_m_x)
        lrmultilply_J!(pad.K.colptr, pad.K.rowval, pad.K.nzval, pad.D, id.n_cols)
    end
    return 0
end

function lrmultilply_J!(J_colptr, J_rowval, J_nzval, v, n_cols)
    T = eltype(v)
    @inbounds @simd for i=1:n_cols
        for idx_row=J_colptr[i]: J_colptr[i+1]-1
                J_nzval[idx_row] *= v[i] * v[J_rowval[idx_row]]
        end
    end

    n = length(J_colptr)
    @inbounds @simd for i=n_cols+1:n-1
        for idx_row= J_colptr[i]: J_colptr[i+1] - 1
            if J_rowval[idx_row] <= n_cols
                J_nzval[idx_row] *= v[J_rowval[idx_row]] # multiply row i by v[i]
            end
        end
    end
end

# function lrmultiply_J2!(J, v)
#     lmul!(v, J)
#     rmul!(J, v)
# end


# iteration functions for the K2.5 system
function factorize_K2_5!(K, K_fact, D, diag_Q , diagind_K, regu, s_l, s_u, x_m_lvar, uvar_m_x, 
                         ilow, iupp, n_rows, n_cols, cnts, qp, T, T0) 

    if regu.regul == :classic
        D .= -regu.ρ
        K.nzval[view(diagind_K, n_cols+1:n_rows+n_cols)] .= regu.δ
    else
        D .= zero(T)
    end
    D[ilow] .-= s_l ./ x_m_lvar
    D[iupp] .-= s_u ./ uvar_m_x
    D[diag_Q.nzind] .-= diag_Q.nzval
    K.nzval[view(diagind_K,1:n_cols)] = D 

    D .= one(T)
    D[ilow] .*= sqrt.(x_m_lvar)
    D[iupp] .*= sqrt.(uvar_m_x)
    lrmultilply_J!(K.colptr, K.rowval, K.nzval, D, n_cols)

    if regu.regul == :dynamic
        # Amax = @views norm(K.nzval[diagind_K], Inf)
        Amax = minimum(D)
        if Amax < sqrt(eps(T)) && cnts.c_pdd < 8
            if T == Float32
                # restore J for next iteration
                D .= one(T) 
                D[ilow] ./= sqrt.(x_m_lvar)
                D[iupp] ./= sqrt.(uvar_m_x)
                lrmultilply_J!(K.colptr, K.rowval, K.nzval, D, n_cols)
                return one(Int) # update to Float64
            elseif qp || cnts.c_pdd < 4
                cnts.c_pdd += 1
                regu.δ /= 10
                K_fact.r2 = max(sqrt(Amax), regu.δ)
                # regu.ρ /= 10
            end
        end
        K_fact.tol = min(Amax, T(eps(T)))
        K_fact = ldl_factorize!(Symmetric(K, :U), K_fact)

    elseif regu.regul == :classic
        K_fact = try ldl_factorize!(Symmetric(K, :U), K_fact)
        catch
            out = update_regu_trycatch!(regu, cnts, T, T0)
            if out == 1 
                # restore J for next iteration
                D .= one(T) 
                D[ilow] ./= sqrt.(x_m_lvar)
                D[iupp] ./= sqrt.(uvar_m_x)
                lrmultilply_J!(K.colptr, K.rowval, K.nzval, D, n_cols)
                return out
            end
            cnts.c_catch += 1
            D .= -regu.ρ       
            D[ilow] .-= s_l ./ x_m_lvar
            D[iupp] .-= s_u ./ uvar_m_x
            D[diag_Q.nzind] .-= diag_Q.nzval
            K.nzval[view(diagind_K,1:n_cols)] = D 
        
            D .= one(T)
            D[ilow] .*= sqrt.(x_m_lvar)
            D[iupp] .*= sqrt.(uvar_m_x)
            K.nzval[view(diagind_K,1:n_cols)] .*= D.^2
            K.nzval[view(diagind_K, n_cols+1:n_rows+n_cols)] .= regu.δ
            K_fact = ldl_factorize!(Symmetric(K, :U), K_fact)
        end

    else # no regularization
        K_fact = ldl_factorize!(Symmetric(K, :U), K_fact)
    end

    cnts.c_catch >= 4 && return 1

    return 0 # factorization succeeded
end

