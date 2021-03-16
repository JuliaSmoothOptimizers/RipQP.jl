# Formulation K2: (if regul==:classic, adds additional regularization parmeters -ρ (top left) and δ (bottom right))
# [-Q - D     A' ] [x] = rhs
# [ A         0  ] [y]

mutable struct PreallocatedData_K2{T<:Real} <: PreallocatedData{T} 
    D                :: Vector{T}                                        # temporary top-left diagonal
    regu             :: Regularization{T}
    diag_Q           :: SparseVector{T,Int} # Q diagonal
    K                :: SparseMatrixCSC{T,Int} # augmented matrix 
    K_fact           :: LDLFactorizations.LDLFactorization{T,Int,Int,Int} # factorized matrix
    fact_fail        :: Bool # true if factorization failed 
    diagind_K        :: Vector{Int} # diagonal indices of J
end

# outer constructor
function PreallocatedData_K2(fd :: QM_FloatData{T}, id :: QM_IntData, iconf :: InputConfig{Tconf}) where {T<:Real, Tconf<:Real}

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

    return PreallocatedData_K2(D,
                               regu,
                               diag_Q, #diag_Q
                               K, #K
                               K_fact, #K_fact
                               false,
                               diagind_K #diagind_K
                               )
end
                
convert(::Type{<:PreallocatedData{T}}, pad :: PreallocatedData_K2{T0}) where {T<:Real, T0<:Real} = 
    PreallocatedData_K2(convert(Array{T}, pad.D),
                        convert(Regularization{T}, pad.regu),
                        convert(SparseVector{T,Int}, pad.diag_Q),
                        convert(SparseMatrixCSC{T,Int}, pad.K),
                        convertldl(T, pad.K_fact),
                        pad.fact_fail,
                        pad.diagind_K
                        )

# function used to solve problems
# solver LDLFactorization
function solver!(pt :: Point{T}, itd :: IterData{T}, fd :: Abstract_QM_FloatData{T}, id :: QM_IntData, res :: Residuals{T}, 
                 dda :: DescentDirectionAllocs{T}, pad :: PreallocatedData_K2{T}, cnts :: Counters, T0 :: DataType, 
                 step :: Symbol) where {T<:Real}
    
    if step == :init # only for starting points
        ldiv!(pad.K_fact, itd.Δxy)
    elseif step == :aff # affine predictor step
        out = factorize_K2!(pad.K, pad.K_fact, pad.D, pad.diag_Q, pad.diagind_K, pad.regu, 
                            pt.s_l, pt.s_u, itd.x_m_lvar, itd.uvar_m_x, id.ilow, id.iupp, 
                            id.n_rows, id.n_cols, cnts, itd.qp, T, T0)
        
        if out == 1 
            pad.fact_fail = true
            return out
        end
        ldiv!(pad.K_fact, dda.Δxy_aff) 
    else # corrector-centering step
        ldiv!(pad.K_fact, itd.Δxy)
        if pad.regu.regul == :classic  # update ρ and δ values, check K diag magnitude 
            out = update_regu_diagJ!(pad.regu, pad.K.nzval, pad.diagind_K, id.n_cols, itd.pdd, 
                                     itd.l_pdd, itd.mean_pdd, cnts, T, T0) 
            out == 1 && return out
        end
    end
    return 0
end

# Init functions for the K2 system
function fill_K2!(K_colptr, K_rowval, K_nzval, D, Q_colptr, Q_rowval, Q_nzval,
                  AT_colptr, AT_rowval, AT_nzval, diag_Q_nzind, δ, n_rows, n_cols, regul)

    added_coeffs_diag = 0 # we add coefficients that do not appear in Q in position i,i if Q[i,i] = 0
    n_nz = length(diag_Q_nzind)
    c_nz = n_nz > 0 ? 1 : 0
    @inbounds for j=1:n_cols  # Q coeffs, tmp diag coefs. 
        K_colptr[j+1] = Q_colptr[j+1] + added_coeffs_diag 
        for k=Q_colptr[j]:(Q_colptr[j+1]-2)
            nz_idx = k + added_coeffs_diag 
            K_rowval[nz_idx] = Q_rowval[k]
            K_nzval[nz_idx] = -Q_nzval[k]
        end
        if c_nz == 0 || c_nz > n_nz || diag_Q_nzind[c_nz] != j
            added_coeffs_diag += 1
            K_colptr[j+1] += 1
            nz_idx = K_colptr[j+1] - 1
            K_rowval[nz_idx] = j
            K_nzval[nz_idx] = D[j]
        else
            if c_nz != 0  
                c_nz += 1
            end
            nz_idx = K_colptr[j+1] - 1
            K_rowval[nz_idx] = j
            K_nzval[nz_idx] = D[j] - Q_nzval[Q_colptr[j+1]-1] 
        end
    end

    countsum = K_colptr[n_cols+1] # current value of K_colptr[Q.n+j+1]
    nnz_top_left = countsum # number of coefficients + 1 already added
    @inbounds for j=1:n_rows
        countsum += AT_colptr[j+1] - AT_colptr[j] 
        if regul == :classic 
            countsum += 1
        end
        K_colptr[n_cols+j+1] = countsum
        for k=AT_colptr[j]:(AT_colptr[j+1]-1)
            nz_idx = regul == :classic ? k + nnz_top_left + j - 2 : k + nnz_top_left - 1
            K_rowval[nz_idx] = AT_rowval[k]
            K_nzval[nz_idx] = AT_nzval[k]
        end
        if regul == :classic
            nz_idx = K_colptr[n_cols+j+1] - 1
            K_rowval[nz_idx] = n_cols + j 
            K_nzval[nz_idx] = δ
        end
    end   
end

function create_K2(id, D, Q, AT, diag_Q, regu)
    # for classic regul only
    n_nz = length(D) - length(diag_Q.nzind) + length(AT.nzval) + length(Q.nzval) 
    T = eltype(D)
    if regu.regul == :classic
        n_nz += id.n_rows
    end
    K_colptr = Vector{Int}(undef, id.n_rows+id.n_cols+1) 
    K_colptr[1] = 1
    K_rowval = Vector{Int}(undef, n_nz)
    K_nzval = Vector{T}(undef, n_nz)
    # [-Q -D    AT]
    # [0        δI]

    fill_K2!(K_colptr, K_rowval, K_nzval, D, Q.colptr, Q.rowval, Q.nzval,
             AT.colptr, AT.rowval, AT.nzval, diag_Q.nzind, regu.δ, id.n_rows, id.n_cols, regu.regul)

    return SparseMatrixCSC(id.n_rows+id.n_cols, id.n_rows+id.n_cols,
                           K_colptr, K_rowval, K_nzval)
end

# iteration functions for the K2 system
function factorize_K2!(K, K_fact, D, diag_Q , diagind_K, regu, s_l, s_u, x_m_lvar, uvar_m_x, 
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

    if regu.regul == :dynamic
        Amax = @views norm(K.nzval[diagind_K], Inf)
        if Amax > T(1e6) / K_fact.r2 && cnts.c_pdd < 8
            if T == Float32
                return one(Int) # update to Float64
            elseif qp || cnts.c_pdd < 4
                cnts.c_pdd += 1
                regu.δ /= 10 
                K_fact.r2 = regu.δ
            end
        end
        K_fact.tol = Amax * T(eps(T))
        K_fact = ldl_factorize!(Symmetric(K, :U), K_fact)

    elseif regu.regul == :classic
        K_fact = try ldl_factorize!(Symmetric(K, :U), K_fact)
        catch
            out = update_regu_trycatch!(regu, cnts, T, T0)
            out == 1 && return out
            cnts.c_catch += 1
            D .= -regu.ρ
            D[ilow] .-= s_l ./ x_m_lvar
            D[iupp] .-= s_u ./ uvar_m_x
            D[diag_Q.nzind] .-= diag_Q.nzval
            K.nzval[view(diagind_K,1:n_cols)] = D 
            K.nzval[view(diagind_K, n_cols+1:n_rows+n_cols)] .= regu.δ
            K_fact = ldl_factorize!(Symmetric(K, :U), K_fact)
        end

    else # no Regularization
        K_fact = ldl_factorize!(Symmetric(K, :U), K_fact)
    end

    cnts.c_catch >= 4 && return 1

    return 0 # factorization succeeded
end

