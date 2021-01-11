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

function mul_ATλ!(ATλ, A, λ)
    ATλ .= zero(eltype(ATλ))
    @inbounds @simd for j=1:A.n
        for k=A.colptr[j]:(A.colptr[j+1]-1)
            i = A.rowval[k]
            ATλ[j] += A.nzval[k] * λ[i]
        end
    end
    return ATλ
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
        colptri = Q.colptr[j]
        i = Q.rowval[colptri]
        if i==j
            diagval[j] = Q.nzval[colptri]
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

function create_J_augm2(IntData, tmp_diag, Q, A, regu, T)
    n_nz =  length(A.nzval) + length(Q.nzval)
    J_augm_colptr = zeros(Int, IntData.n_rows+IntData.n_cols+1)
    J_augm_rowval = zeros(Int, n_nz)
    J_augm_nzval = zeros(T, n_nz)
    for k in 1:length(Q.nzval) # add +1 in col j if j=Q.rowval[k]
        J_augm_colptr[Q.rowval[k]+1] += 1
    end
    for k in 1:length(A.nzval)
        J_augm_colptr[Q.m+A.rowval[k]+1] += 1
    end
    countsum = 1
    J_augm_colptr[1] = countsum
    for k in 2:length(J_augm_colptr)
        overwritten = J_augm_colptr[k]
        J_augm_colptr[k] = countsum
        countsum += overwritten
    end
    for i in 1:Q.n
        for k in nzrange(Q, i)
            j = Q.rowval[k]
            colj = J_augm_colptr[j+1]
            J_augm_rowval[colj] = i
            J_augm_nzval[colj] = -Q.nzval[k]
            J_augm_colptr[j+1] += 1
        end
    end
    for i in 1:A.n
        for k in nzrange(A, i)
            j = IntData.n_cols+A.rowval[k]
            colj = J_augm_colptr[j+1]
            J_augm_rowval[colj] = i
            J_augm_nzval[colj] = A.nzval[k]
            J_augm_colptr[j+1] += 1
        end
    end
    J_augm = SparseMatrixCSC(IntData.n_rows+IntData.n_cols, IntData.n_rows+IntData.n_cols,
                             J_augm_colptr, J_augm_rowval, J_augm_nzval)
    diagind_J = diagind(J_augm)
    J_augm[diagind_J[1:IntData.n_cols]] = tmp_diag
    if regu.regul == :classic
        J_augm[diagind_J[IntData.n_cols+1:end]] .= regu.δ
    end
    # J_augm = [Q                                            A';
    #           spzeros(T, IntData.n_rows, IntData.n_cols)     regu.δ.*I]
    # J_augm = spzeros(T, IntData.n_rows+IntData.n_cols, IntData.n_rows+IntData.n_cols)
    # J_augm[1:IntData.n_cols, 1:IntData.n_cols] = Q'
    # J_augm.nzval .= .-J_augm.nzval
    # diagind_J = diagind(J_augm)
    # J_augm[view(diagind_J, 1:IntData.n_cols)] .+= tmp_diag
    # J_augm[1:IntData.n_cols, IntData.n_cols+1:end] = A'
    # J_augm[diagind_J[IntData.n_cols+1:end]] .= regu.δ
    return J_augm
end

function create_J_augm3(IntData, tmp_diag, Q, A, regu, diag_Q, T)
    # for classic regul only
    n_nz = length(tmp_diag) - length(diag_Q.nzind) + length(A.nzval) + length(Q.nzval) + IntData.n_rows
    J_augm_colptr = ones(Int, IntData.n_rows+IntData.n_cols+1)
    J_augm_rowval = zeros(Int, n_nz)
    J_augm_nzval = zeros(T, n_nz)
    # [-Q -tmp_diag    0 ]
    # [A               δI]
    c_Q = 1
    c_A = 1
    for j=1:Q.m # top and bottom left blocs
        colptr_j = J_augm_colptr[j]
        diag_idx = 0
        # add Q coeffs
        diag_zero = diag_Q[j] == zero(T) ? 1 : 0 # add 1 if Q[j,j] = 0 because of tmp_diag
        J_augm_colptr[j+1] = colptr_j + Q.colptr[j+1] - Q.colptr[j] + diag_zero
        J_augm_rowval[colptr_j] = j # diagonal 1st coef
        J_augm_nzval[colptr_j] = tmp_diag[j]
        for k=colptr_j+diag_zero:(J_augm_colptr[j+1]-1)
            J_augm_rowval[k] = Q.rowval[c_Q]
            J_augm_nzval[k] += Q.nzval[c_Q]
            c_Q += 1
        end
        # add A coeffs
        println("ok")
        prev_colptr = J_augm_colptr[j+1]
        J_augm_colptr[j+1] += A.colptr[j+1] - A.colptr[j]
        for k=prev_colptr:(J_augm_colptr[j+1]-1)
            J_augm_rowval[k] = IntData.n_cols + A.rowval[c_A]
            J_augm_nzval[k] = A.nzval[c_A]
            c_A += 1
        end
    end
    # bottom right bloc
    for j=Q.m+1:length(J_augm_colptr)-1
        colptr_j = J_augm_colptr[j]
        J_augm_colptr[j+1] = colptr_j + 1
        J_augm_rowval[colptr_j] = j
        J_augm_nzval[colptr_j] = regu.δ
    end
    J_augm = SparseMatrixCSC(IntData.n_rows+IntData.n_cols, IntData.n_rows+IntData.n_cols,
                             J_augm_colptr, J_augm_rowval, J_augm_nzval)
    return
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
