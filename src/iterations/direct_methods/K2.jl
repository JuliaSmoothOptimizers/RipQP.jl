# Formulation K2: (if regul==:classic, adds additional regularization parmeters -ρ (top left) and δ (bottom right))
# [-Q - tmp_diag     A' ] [x] = rhs
# [ A                0  ] [y]

# iter_data type for the K2 formulation
mutable struct iter_data_K2{T<:Real} <: iter_data{T} 
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
    qp          :: Bool # true if qp false if lp
end

convert(::Type{<:iter_data{T}}, itd :: iter_data_K2{T0}) where {T<:Real, T0<:Real} = 
    iter_data_K2(convert(Array{T}, itd.tmp_diag),
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
                 convert(T, itd.mean_pdd),
                 itd.qp
                 )

mutable struct preallocated_data_K2{T<:Real} <: preallocated_data{T} 
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
                
convert(::Type{<:preallocated_data{T}}, pad :: preallocated_data_K2{T0}) where {T<:Real, T0<:Real} = 
    preallocated_data_K2(convert(Array{T}, pad.Δxy_aff),
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

# Init functions for the K2 system
function fill_K2!(J_augm_colptr, J_augm_rowval, J_augm_nzval, tmp_diag, Q_colptr, Q_rowval, Q_nzval,
                  AT_colptr, AT_rowval, AT_nzval, diag_Q_nzind, δ, n_rows, n_cols, regul)

    added_coeffs_diag = 0 # we add coefficients that do not appear in Q in position i,i if Q[i,i] = 0
    n_nz = length(diag_Q_nzind)
    c_nz = n_nz > 0 ? 1 : 0
    @inbounds for j=1:n_cols  # Q coeffs, tmp diag coefs. 
        J_augm_colptr[j+1] = Q_colptr[j+1] + added_coeffs_diag 
        for k=Q_colptr[j]:(Q_colptr[j+1]-2)
            nz_idx = k + added_coeffs_diag 
            J_augm_rowval[nz_idx] = Q_rowval[k]
            J_augm_nzval[nz_idx] = -Q_nzval[k]
        end
        if c_nz == 0 || c_nz > n_nz || diag_Q_nzind[c_nz] != j
            added_coeffs_diag += 1
            J_augm_colptr[j+1] += 1
            nz_idx = J_augm_colptr[j+1] - 1
            J_augm_rowval[nz_idx] = j
            J_augm_nzval[nz_idx] = tmp_diag[j]
        else
            if c_nz != 0  
                c_nz += 1
            end
            nz_idx = J_augm_colptr[j+1] - 1
            J_augm_rowval[nz_idx] = j
            J_augm_nzval[nz_idx] = tmp_diag[j] - Q_nzval[Q_colptr[j+1]-1] 
        end
    end

    countsum = J_augm_colptr[n_cols+1] # current value of J_augm_colptr[Q.n+j+1]
    nnz_top_left = countsum # number of coefficients + 1 already added
    @inbounds for j=1:n_rows
        countsum += AT_colptr[j+1] - AT_colptr[j] 
        if regul == :classic 
            countsum += 1
        end
        J_augm_colptr[n_cols+j+1] = countsum
        for k=AT_colptr[j]:(AT_colptr[j+1]-1)
            nz_idx = regul == :classic ? k + nnz_top_left + j - 2 : k + nnz_top_left - 1
            J_augm_rowval[nz_idx] = AT_rowval[k]
            J_augm_nzval[nz_idx] = AT_nzval[k]
        end
        if regul == :classic
            nz_idx = J_augm_colptr[n_cols+j+1] - 1
            J_augm_rowval[nz_idx] = n_cols + j 
            J_augm_nzval[nz_idx] = δ
        end
    end   
end

function create_K2(id, tmp_diag, Q, AT, diag_Q, regu)
    # for classic regul only
    n_nz = length(tmp_diag) - length(diag_Q.nzind) + length(AT.nzval) + length(Q.nzval) 
    T = eltype(tmp_diag)
    if regu.regul == :classic
        n_nz += id.n_rows
    end
    J_augm_colptr = Vector{Int}(undef, id.n_rows+id.n_cols+1) 
    J_augm_colptr[1] = 1
    J_augm_rowval = Vector{Int}(undef, n_nz)
    J_augm_nzval = Vector{T}(undef, n_nz)
    # [-Q -tmp_diag    AT]
    # [0               δI]

    fill_K2!(J_augm_colptr, J_augm_rowval, J_augm_nzval, tmp_diag, Q.colptr, Q.rowval, Q.nzval,
             AT.colptr, AT.rowval, AT.nzval, diag_Q.nzind, regu.δ, id.n_rows, id.n_cols, regu.regul)

    return SparseMatrixCSC(id.n_rows+id.n_cols, id.n_rows+id.n_cols,
                           J_augm_colptr, J_augm_rowval, J_augm_nzval)
end

function create_iterdata_K2(fd :: QM_FloatData{T}, id :: QM_IntData, mode :: Symbol, 
                            regul :: Symbol) where {T<:Real}
    # init regularization values
    if mode == :mono
        regu = regularization(T(sqrt(eps())*1e5), T(sqrt(eps())*1e5), 1e-5*sqrt(eps(T)), 1e0*sqrt(eps(T)), regul)
        tmp_diag = -T(1.0e0)/2 .* ones(T, id.n_cols)
    else
        regu = regularization(T(sqrt(eps())*1e5), T(sqrt(eps())*1e5), T(sqrt(eps(T))*1e0), T(sqrt(eps(T))*1e0), regul)
        tmp_diag = -T(1.0e-2) .* ones(T, id.n_cols)
    end
    diag_Q = get_diag_Q(fd.Q.colptr, fd.Q.rowval, fd.Q.nzval, id.n_cols)
    J_augm = create_K2(id, tmp_diag, fd.Q, fd.AT, diag_Q, regu)

    diagind_J = get_diag_sparseCSC(J_augm.colptr, id.n_rows+id.n_cols)
    J_fact = ldl_analyze(Symmetric(J_augm, :U))
    if regu.regul == :dynamic
        Amax = @views norm(J_augm.nzval[diagind_J], Inf)
        J_fact.r1 = T(-eps(T)^(3/4))
        J_fact.r2 = T(sqrt(eps(T)))
        J_fact.tol = Amax*T(eps(T))
        J_fact.n_d = id.n_cols
        J_fact = ldl_factorize!(Symmetric(J_augm, :U), J_fact)
    end
    J_fact = ldl_factorize!(Symmetric(J_augm, :U), J_fact)
    J_fact.__factorized = true
    itd = iter_data_K2(tmp_diag, # tmp diag
                       diag_Q, #diag_Q
                       J_augm, #J_augm
                       J_fact, #J_fact
                       diagind_J, #diagind_J
                       regu,
                       zeros(T, id.n_low), # x_m_lvar
                       zeros(T, id.n_upp), # uvar_m_x
                       zeros(T, id.n_cols), # init Qx
                       zeros(T, id.n_cols), # init ATy
                       zeros(T, id.n_rows), # Ax
                       zero(T), #xTQx
                       zero(T), #cTx
                       zero(T), #pri_obj
                       zero(T), #dual_obj
                       zero(T), #μ
                       zero(T),#pdd
                       zeros(T, 6), #l_pdd
                       one(T), #mean_pdd
                       nnz(fd.Q) > 0
                       )
    
    
    pad = preallocated_data_K2(zeros(T, id.n_cols+id.n_rows), # Δxy_aff
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
                               zeros(T, id.n_upp) #rxs_u
                               )
    
    # init system
    # solve [-Q-tmp_diag    A' ] [x] = [b]  to initialize (x, y, s_l, s_u)
    #       [      A         0 ] [y] = [0]
    pad.Δxy[id.n_cols+1: end] = fd.b
    ldiv!(itd.J_fact, pad.Δxy)
    pt0 = point(pad.Δxy[1:id.n_cols], pad.Δxy[id.n_cols+1:end], zeros(T, id.n_low), zeros(T, id.n_upp))

    return itd, pad, pt0
end

# iteration functions for the K2 system
function factorize_K2!(J_augm, J_fact, tmp_diag, diag_Q , diagind_J, regu, s_l, s_u, x_m_lvar, uvar_m_x, 
                       ilow, iupp, n_rows, n_cols, cnts, qp, T, T0) 

    if regu.regul == :classic
        tmp_diag .= -regu.ρ
        J_augm.nzval[view(diagind_J, n_cols+1:n_rows+n_cols)] .= regu.δ
    else
        tmp_diag .= zero(T)
    end
    tmp_diag[ilow] .-= s_l ./ x_m_lvar
    tmp_diag[iupp] .-= s_u ./ uvar_m_x
    tmp_diag[diag_Q.nzind] .-= diag_Q.nzval
    J_augm.nzval[view(diagind_J,1:n_cols)] = tmp_diag 

    if regu.regul == :dynamic
        Amax = @views norm(J_augm.nzval[diagind_J], Inf)
        if Amax > T(1e6) / J_fact.r2 && cnts.c_pdd < 8
            if T == Float32
                return one(Int) # update to Float64
            elseif qp || cnts.c_pdd < 4
                cnts.c_pdd += 1
                regu.δ /= 10 
                J_fact.r2 = regu.δ
            end
        end
        J_fact.tol = Amax * T(eps(T))
        J_fact = ldl_factorize!(Symmetric(J_augm, :U), J_fact)

    elseif regu.regul == :classic
        J_fact = try ldl_factorize!(Symmetric(J_augm, :U), J_fact)
        catch
            out = update_regu_trycatch!(regu, cnts, T, T0)
            out == 1 && return out
            cnts.c_catch += 1
            tmp_diag .= -regu.ρ
            tmp_diag[ilow] .-= s_l ./ x_m_lvar
            tmp_diag[iupp] .-= s_u ./ uvar_m_x
            tmp_diag[diag_Q.nzind] .-= diag_Q.nzval
            J_augm.nzval[view(diagind_J,1:n_cols)] = tmp_diag 
            J_augm.nzval[view(diagind_J, n_cols+1:n_rows+n_cols)] .= regu.δ
            J_fact = ldl_factorize!(Symmetric(J_augm, :U), J_fact)
        end

    else # no regularization
        J_fact = ldl_factorize!(Symmetric(J_augm, :U), J_fact)
    end

    cnts.c_catch >= 4 && return 1

    return 0 # factorization succeeded
end

function solve_K2!(pt, itd, fd, id, res, pad, cnts, T, T0)

    out = factorize_K2!(itd.J_augm, itd.J_fact, itd.tmp_diag, itd.diag_Q, itd.diagind_J, itd.regu, 
                        pt.s_l, pt.s_u, itd.x_m_lvar, itd.uvar_m_x, id.ilow, id.iupp, 
                        id.n_rows, id.n_cols, cnts, itd.qp, T, T0)

    out == 1 && return out

    # affine scaling step
    solve_augmented_system_aff!(pad.Δxy_aff, pad.Δs_l_aff, pad.Δs_u_aff, itd.J_fact, res.rc, res.rb, 
                                itd.x_m_lvar, itd.uvar_m_x, pt.s_l, pt.s_u, id.ilow, id.iupp, id.n_cols)
    α_aff_pri, α_aff_dual = compute_αs(pt.x, pt.s_l, pt.s_u, fd.lvar, fd.uvar, pad.Δxy_aff, pad.Δs_l_aff, 
                                       pad.Δs_u_aff, id.n_cols)
    # (x-lvar, uvar-x, s_l, s_u) .+= α_aff * Δ_aff                                 
    update_pt_aff!(pad.x_m_l_αΔ_aff, pad.u_m_x_αΔ_aff, pad.s_l_αΔ_aff, pad.s_u_αΔ_aff, pad.Δxy_aff, 
                   pad.Δs_l_aff, pad.Δs_u_aff, itd.x_m_lvar, itd.uvar_m_x, pt.s_l, pt.s_u, 
                   α_aff_pri, α_aff_dual, id.ilow, id.iupp)
    μ_aff = compute_μ(pad.x_m_l_αΔ_aff, pad.u_m_x_αΔ_aff, pad.s_l_αΔ_aff, pad.s_u_αΔ_aff, id.n_low, id.n_upp)
    σ = (μ_aff / itd.μ)^3

    # corrector-centering step
    pad.rxs_l .= @views -σ * itd.μ .+ pad.Δxy_aff[id.ilow] .* pad.Δs_l_aff
    pad.rxs_u .= @views σ * itd.μ .+ pad.Δxy_aff[id.iupp] .* pad.Δs_u_aff
    solve_augmented_system_cc!(itd.J_fact, pad.Δxy, pad.Δs_l, pad.Δs_u, itd.x_m_lvar, itd.uvar_m_x, 
                               pad.rxs_l, pad.rxs_u, pt.s_l, pt.s_u, id.ilow, id.iupp)

    # final direction
    pad.Δxy .+= pad.Δxy_aff  
    pad.Δs_l .+= pad.Δs_l_aff
    pad.Δs_u .+= pad.Δs_u_aff

    # update regularization
    if itd.regu.regul == :classic  # update ρ and δ values, check J_augm diag magnitude 
        out = update_regu_diagJ!(itd.regu, itd.J_augm.nzval, itd.diagind_J, id.n_cols, itd.pdd, 
                                 itd.l_pdd, itd.mean_pdd, cnts, T, T0) 
    end
    
    return out
end
