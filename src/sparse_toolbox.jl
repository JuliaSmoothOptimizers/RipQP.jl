function mul_Qx!(Qx, Q, x)
    # right mutiplication for sparse CSC symetric matrix
    Qx .= zero(eltype(Qx))
    @inbounds @simd for j=1:Q.m
        for k=Q.colptr[j]:(Q.colptr[j+1]-1)
            i = Q.rowval[k]
            Qx[i] += Q.nzval[k] * x[j]
            if j != Q.rowval[k]
                Qx[j] += Q.nzval[k]*x[i]
            end
        end
    end
    return Qx
end

function mul_Aλ!(Aλ, A, λ)
    Aλ .= zero(eltype(Aλ))
    @inbounds @simd for j=1:A.n
        for k=A.colptr[j]:(A.colptr[j+1]-1)
            i = A.rowval[k]
            Aλ[j] += A.nzval[k] * λ[i]
        end
    end
    return Aλ
end

function mul_Ax!(Ax, A, x)
    Ax .= zero(eltype(Ax))
    @inbounds @simd for j=1:A.n
        for k=A.colptr[j]:(A.colptr[j+1]-1)
            i = A.rowval[k]
            Ax[i] += A.nzval[k] * x[j]
        end
    end
    return Ax
end

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

function get_diag_Q(Q) # lower triangular
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

function create_J_augm(IntData, tmp_diag, Qvals, Avals, regu, T)
    if regu.regul == :classic
        J_augmrows = vcat(IntData.Qcols, IntData.Acols, IntData.n_cols+1:IntData.n_cols+IntData.n_rows,
                          1:IntData.n_cols)
        J_augmcols = vcat(IntData.Qrows, IntData.Arows.+IntData.n_cols,
                          IntData.n_cols+1:IntData.n_cols+IntData.n_rows, 1:IntData.n_cols)
        J_augmvals = vcat(.-Qvals, Avals, regu.δ.*ones(T, IntData.n_rows), tmp_diag)
    else
        J_augmrows = vcat(IntData.Qcols, IntData.Acols, 1:IntData.n_cols)
        J_augmcols = vcat(IntData.Qrows, IntData.Arows.+IntData.n_cols, 1:IntData.n_cols)
        J_augmvals = vcat(.-Qvals, Avals, tmp_diag)
    end
    J_augm = sparse(J_augmrows, J_augmcols, J_augmvals,
                    IntData.n_rows+IntData.n_cols, IntData.n_rows+IntData.n_cols)
    return J_augm
end

function create_J_augm2(IntData, tmp_diag, Q, A, diag_Q, regu, T)
    n_nz =  length(A.nzval) + length(Q.nzval) + IntData.n_cols - nnz(diag_Q) + IntData.n_rows
    J_augm_colptr = zeros(Int, IntData.n_rows+IntData.n_cols+1)
    J_augm_rowval = zeros(Int, n_nz)
    J_augm_nzval = zeros(T, n_nz) 
    # regul classic
    @inbounds @simd for k in 1:length(Q.nzval) # add +1 in col j if j=Q.rowval[k]
        J_augm_colptr[Q.rowval[k]+1] += 1
    end
    @inbounds @simd for i=1:Q.m
        if diag_Q[i] == zero(T)
            J_augm_colptr[i+1] += 1
        end
    end
    @inbounds @simd for k=1:length(A.nzval)
        J_augm_colptr[Q.m+A.rowval[k]+1] += 1
    end
    J_augm_colptr[Q.m+2:end] .+= 1
    countsum = 1
    J_augm_colptr[1] = countsum
    @inbounds for k=2:length(J_augm_colptr) # J_augm_colptr shifted forward one position
        overwritten = J_augm_colptr[k]
        J_augm_colptr[k] = countsum
        countsum += overwritten
    end

    count_zeros_Q = 0
    @inbounds for i=1:Q.n
        if diag_Q[i] == zero(T)
            count_zeros_Q += 1
            coli = J_augm_colptr[i+1]
            J_augm_rowval[coli] = i
            J_augm_nzval[coli] = tmp_diag[i] 
            J_augm_colptr[i+1] += 1
        end
        for k in nzrange(Q, i)
            j = Q.rowval[k]
            val = Q.nzval[k]
            colj = J_augm_colptr[j+1] 
            J_augm_rowval[colj] = i
            if i == j
                J_augm_nzval[colj] = tmp_diag[i] - val
            else
                J_augm_nzval[colj] = -val
            end
            J_augm_colptr[j+1] += 1
        end
        for k in nzrange(A, i)
            j = A.rowval[k] + IntData.n_cols
            val = A.nzval[k]
            colj = J_augm_colptr[j+1] 
            J_augm_rowval[colj] = i
            J_augm_nzval[colj] = val
            J_augm_colptr[j+1] += 1
        end
    end

    @inbounds for i=IntData.n_cols+1:IntData.n_cols+IntData.n_rows
        colj = J_augm_colptr[i+1]
        J_augm_rowval[colj] = i
        J_augm_nzval[colj] = regu.δ
        J_augm_colptr[i+1] += 1
    end 

    J_augm = SparseMatrixCSC(IntData.n_rows+IntData.n_cols, IntData.n_rows+IntData.n_cols,
                             J_augm_colptr, J_augm_rowval, J_augm_nzval)
    return J_augm
end

function create_J_augm3(IntData, tmp_diag, Q, A, diag_Q, regu, T)
    # # for classic regul only
    # n_nz = length(tmp_diag) - length(diag_Q.nzind) + length(A.nzval) + length(Q.nzval) + IntData.n_rows
    # J_augm_colptr = Vector{Int}(undef, IntData.n_rows+IntData.n_cols+1) 
    # J_augm_colptr[1] = 1
    # J_augm_rowval = Vector{Int}(undef, n_nz)
    # J_augm_nzval =  Vector{T}(undef, n_nz)
    # # [-Q -tmp_diag    A ]
    # # [0               δI]

    # added_coeffs_diag = 0 # we add coefficients that do not appear in Q in position i,i if Q[i,i] = 0
    # @inbounds for j=1:Q.n  # Q coeffs, tmp diag coefs. 
    #     J_augm_colptr[j+1] = Q.colptr[j+1] + added_coeffs_diag 
    #     for k=Q.colptr[j]:(Q.colptr[j+1]-2)
    #         nz_idx = k + added_coeffs_diag 
    #         J_augm_rowval[nz_idx] = Q.rowval[k]
    #         J_augm_nzval[nz_idx] = Q.nzval[k]
    #     end
    #     if diag_Q[j] == zero(T)
    #         added_coeffs_diag += 1
    #         J_augm_colptr[j+1] += 1
    #         nz_idx = J_augm_colptr[j+1] - 1
    #         J_augm_rowval[nz_idx] = j
    #         J_augm_nzval[nz_idx] = tmp_diag[j]
    #     else
    #         nz_idx = J_augm_colptr[j+1] - 1
    #         J_augm_rowval[nz_idx] = j
    #         J_augm_nzval[nz_idx] = tmp_diag[j] - Q.nzval[Q.colptr[j+1]-1] 
    #     end
    # end

    # countsum = J_augm_colptr[Q.n+1] # current value of J_augm_colptr[Q.n+j+1]
    # nnz_top_left = countsum # number of coefficients + 1 already added
    # @inbounds for j=1:A.n
    #     countsum += A.colptr[j+1] - A.colptr[j] + 1
    #     J_augm_colptr[Q.n+j+1] = countsum
    #     for k in nzrange(A, j)
    #         nz_idx = k + nnz_top_left + j - 2
    #         J_augm_rowval[nz_idx] = A.rowval[k]
    #         J_augm_nzval[nz_idx] = A.nzval[k]
    #     end
    #     nz_idx = J_augm_colptr[Q.n+j+1] - 1
    #     J_augm_rowval[nz_idx] = Q.n + j 
    #     J_augm_nzval[nz_idx] = regu.δ
    # end

    # J_augm = SparseMatrixCSC(IntData.n_rows+IntData.n_cols, IntData.n_rows+IntData.n_cols,
    #                          J_augm_colptr, J_augm_rowval, J_augm_nzval)
    J_augm = [.-Q  A
              spzeros(T, IntData.n_rows, IntData.n_cols+IntData.n_rows)]
    J_augm[diagind(J_augm)[1:IntData.n_cols]] = tmp_diag
    J_augm[diagind(J_augm)[IntData.n_cols+1:IntData.n_rows+IntData.n_cols]] .= regu.δ
    return J_augm
end

function fill_J_augm4!(J_augm_colptr, J_augm_rowval, J_augm_nzval, tmp_diag, Q_colptr, Q_rowval, Q_nzval,
                       A_colptr, A_rowval, A_nzval, diag_Q_nzind, δ, n_rows, n_cols, T)

    added_coeffs_diag = 0 # we add coefficients that do not appear in Q in position i,i if Q[i,i] = 0
    c_nz = length(diag_Q_nzind) > 0 ? 1 : 0
    @inbounds for j=1:n_cols  # Q coeffs, tmp diag coefs. 
        J_augm_colptr[j+1] = Q_colptr[j+1] + added_coeffs_diag 
        for k=Q_colptr[j]:(Q_colptr[j+1]-2)
            nz_idx = k + added_coeffs_diag 
            J_augm_rowval[nz_idx] = Q_rowval[k]
            J_augm_nzval[nz_idx] = Q_nzval[k]
        end
        if c_nz == 0 || diag_Q_nzind[c_nz] != j
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
        countsum += A_colptr[j+1] - A_colptr[j] + 1
        J_augm_colptr[n_cols+j+1] = countsum
        for k=A_colptr[j]:(A_colptr[j+1]-1)
            nz_idx = k + nnz_top_left + j - 2
            J_augm_rowval[nz_idx] = A_rowval[k]
            J_augm_nzval[nz_idx] = A_nzval[k]
        end
        nz_idx = J_augm_colptr[n_cols+j+1] - 1
        J_augm_rowval[nz_idx] = n_cols + j 
        J_augm_nzval[nz_idx] = δ
    end   
end

function create_J_augm4(IntData, tmp_diag, Q, A, diag_Q, regu, T)
    # for classic regul only
    n_nz = length(tmp_diag) - length(diag_Q.nzind) + length(A.nzval) + length(Q.nzval) + IntData.n_rows
    J_augm_colptr = Vector{Int}(undef, IntData.n_rows+IntData.n_cols+1) 
    J_augm_colptr[1] = 1
    J_augm_rowval = Vector{Int}(undef, n_nz)
    J_augm_nzval = Vector{T}(undef, n_nz)
    # [-Q -tmp_diag    A ]
    # [0               δI]

    fill_J_augm4!(J_augm_colptr, J_augm_rowval, J_augm_nzval, tmp_diag, Q.colptr, Q.rowval, Q.nzval,
                  A.colptr, A.rowval, A.nzval, diag_Q.nzind, regu.δ, IntData.n_rows, 
                  IntData.n_cols, T)

    return SparseMatrixCSC(IntData.n_rows+IntData.n_cols, IntData.n_rows+IntData.n_cols,
                           J_augm_colptr, J_augm_rowval, J_augm_nzval)
end

function nb_corrector_steps(J :: SparseMatrixCSC{T,Int}, n_cols :: Int) where {T<:Real}
    # number to determine the number of centrality corrections (Gondzio's procedure)
    Ef, Es, rfs = 0, 16 * n_cols, zero(T) # 14n = ratio tests and vector initializations
    @inbounds @simd for j=1:J.n-1
        lj = (J.colptr[j+1]-J.colptr[j])
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
