# Formulation K2: (if regul==:classic, adds additional Regularization parmeters -ρ (top left) and δ (bottom right))
# [-Q_X - D      sqrt(X1X2)A' ] [̃x] = rhs
# [ A sqrt(X1x2)         0    ] [y]
# where Q_X = sqrt(X1X2) Q sqrt(X1X2) and D = s_l X2 + s_u X1
# and  Δ x = sqrt(X1 X2) Δ ̃x
 
mutable struct PreallocatedData_K2_5{T<:Real} <: PreallocatedData{T} 
    D                :: Vector{T}                                        # temporary top-left diagonal
    regu             :: Regularization{T}
    diag_Q           :: SparseVector{T,Int} # Q diagonal
    K                :: SparseMatrixCSC{T,Int} # augmented matrix 
    K_fact           :: LDLFactorizations.LDLFactorization{T,Int,Int,Int} # factorized matrix
    fact_fail        :: Bool # true if factorization failed 
    diagind_K        :: Vector{Int} # diagonal indices of J
end

function PreallocatedData_K2_5(fd :: QM_FloatData{T}, id :: QM_IntData, iconf :: InputConfig{Tconf}) where {T<:Real, Tconf<:Real} 
    # init Regularization values
    if iconf.mode == :mono
        regu = Regularization(T(sqrt(eps())*1e5), T(sqrt(eps())*1e5), 1e-5*sqrt(eps(T)), 1e0*sqrt(eps(T)), iconf.regul)
        D = -T(1.0e0)/2 .* ones(T, id.n_cols)
    else
        regu = Regularization(T(sqrt(eps())*1e5), T(sqrt(eps())*1e5), T(sqrt(eps(T))*1e0), T(sqrt(eps(T))*1e0), iconf.regul)
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

    return PreallocatedData_K2_5(D,
                                 regu,
                                 diag_Q, #diag_Q
                                 K, #K
                                 K_fact, #K_fact
                                 false,
                                 diagind_K #diagind_K
                                 )
end
                
convert(::Type{<:PreallocatedData{T}}, pad :: PreallocatedData_K2_5{T0}) where {T<:Real, T0<:Real} = 
    PreallocatedData_K2_5(convert(Array{T}, pad.D),
                          convert(Regularization{T}, pad.regu),
                          convert(SparseVector{T,Int}, pad.diag_Q),
                          convert(SparseMatrixCSC{T,Int}, pad.K),
                          convertldl(T, pad.K_fact),
                          pad.fact_fail,
                          pad.diagind_K
                          )

# solver LDLFactorization
function solver!(pt :: Point{T}, itd :: IterData{T}, fd :: Abstract_QM_FloatData{T}, id :: QM_IntData, res :: Residuals{T}, 
                 dda :: DescentDirectionAllocs{T}, pad :: PreallocatedData_K2_5{T}, cnts :: Counters, T0 :: DataType, 
                 step :: Symbol) where {T<:Real} 

    if step == :init 
        LDLFactorizations.ldiv!(pad.K_fact, itd.Δxy)
    elseif step == :aff 
        out = factorize_K2_5!(pad.K, pad.K_fact, pad.D, pad.diag_Q, pad.diagind_K, pad.regu, 
                              pt.s_l, pt.s_u, itd.x_m_lvar, itd.uvar_m_x, id.ilow, id.iupp, 
                              id.n_rows, id.n_cols, cnts, itd.qp, T, T0)
        out == 1 && return out

        dda.Δxy_aff[1:id.n_cols] .*= pad.D
        LDLFactorizations.ldiv!(pad.K_fact, dda.Δxy_aff) 
        dda.Δxy_aff[1:id.n_cols] .*= pad.D
    else
        itd.Δxy[1:id.n_cols] .*= pad.D
        LDLFactorizations.ldiv!(pad.K_fact, itd.Δxy)
        itd.Δxy[1:id.n_cols] .*= pad.D

        if pad.regu.regul == :classic  # update ρ and δ values, check K diag magnitude 
            out = update_regu_diagJ_K2_5!(pad.regu, pad.D, itd.pdd, itd.l_pdd, itd.mean_pdd, cnts, T, T0) 
            out == 1 && return out
        end
    
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

    else # no Regularization
        K_fact = ldl_factorize!(Symmetric(K, :U), K_fact)
    end

    cnts.c_catch >= 4 && return 1

    return 0 # factorization succeeded
end

