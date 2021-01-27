function get_diag_sparseCSC(M; tri=:U)
    # get diagonal index of M.nzval
    # we assume all columns of M are non empty, and M triangular (:L or :U)
    @assert tri ==:U || tri == :L
    T = eltype(M)
    n = length(M.rowval)
    diagind = zeros(Int, M.m) # square matrix
    index = M.rowval[1] # 1
    if tri == :U
        @inbounds @simd for i=1:M.m
            diagind[i] = M.colptr[i+1] - 1
        end
    else
        @inbounds @simd for i=1:M.m
            diagind[i] = M.colptr[i]
        end
    end
    return diagind
end

function get_diag_Q(Q) 
    T = eltype(Q.nzval)
    diagval = spzeros(T, Q.m)
    @inbounds @simd for j=1:Q.m
        for k in nzrange(Q, j)
            if j == Q.rowval[k]
                diagval[j] = Q.nzval[k]
            end
        end
    end
    return diagval
end

function fill_J_augm!(J_augm_colptr, J_augm_rowval, J_augm_nzval, tmp_diag, Q_colptr, Q_rowval, Q_nzval,
                       AT_colptr, AT_rowval, AT_nzval, diag_Q_nzind, δ, n_rows, n_cols, regul, T)

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

function create_J_augm(IntData, tmp_diag, Q, AT, diag_Q, regu, T)
    # for classic regul only
    n_nz = length(tmp_diag) - length(diag_Q.nzind) + length(AT.nzval) + length(Q.nzval) 
    if regu.regul == :classic
        n_nz += IntData.n_rows
    end
    J_augm_colptr = Vector{Int}(undef, IntData.n_rows+IntData.n_cols+1) 
    J_augm_colptr[1] = 1
    J_augm_rowval = Vector{Int}(undef, n_nz)
    J_augm_nzval = Vector{T}(undef, n_nz)
    # [-Q -tmp_diag    AT]
    # [0               δI]

    fill_J_augm!(J_augm_colptr, J_augm_rowval, J_augm_nzval, tmp_diag, Q.colptr, Q.rowval, Q.nzval,
                  AT.colptr, AT.rowval, AT.nzval, diag_Q.nzind, regu.δ, IntData.n_rows, 
                  IntData.n_cols, regu.regul, T)

    return SparseMatrixCSC(IntData.n_rows+IntData.n_cols, IntData.n_rows+IntData.n_cols,
                           J_augm_colptr, J_augm_rowval, J_augm_nzval)
end

function nb_corrector_steps(J_colptr :: Vector{Int}, n_rows :: Int, n_cols :: Int, T) 
    # number to determine the number of centrality corrections (Gondzio's procedure)
    Ef, Es, rfs = 0, 16 * n_cols, zero(T) # 14n = ratio tests and vector initializations
    @inbounds @simd for j=1:n_rows+n_cols
        lj = (J_colptr[j+1]-J_colptr[j])
        Ef += lj^2
        Es += lj
    end
    rfs = T(Ef / Es)
    if rfs <= 10
        K = 0
    elseif 10 < rfs <= 30
        K = 1
    elseif 30 < rfs <= 50
        K = 2
    elseif rfs > 50
        K = 3
    else
        p = Int(rfs / 50)
        K = p + 2
        if K > 10
            K = 10
        end
    end
    return K
end
