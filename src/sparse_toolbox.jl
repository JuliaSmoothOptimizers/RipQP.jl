function mul_Qx_COO!(Qx, Qrows, Qcols, Qvals, x)
    # right mutiplication for sparse COO symetric matrix
    Qx .= zero(eltype(Qx))
    @inbounds @simd for i=1:length(Qcols)
        Qx[Qrows[i]] += Qvals[i] * x[Qcols[i]]
        if Qrows[i] != Qcols[i]
            Qx[Qcols[i]] += Qvals[i]*x[Qrows[i]]
        end
    end
    return Qx
end

function mul_ATλ_COO!(ATλ, Arows, Acols, Avals, λ)
    ATλ .= zero(eltype(ATλ))
    @inbounds @simd for i=1:length(Acols)
        ATλ[Acols[i]] += Avals[i] * λ[Arows[i]]
    end
    return ATλ
end

function mul_Ax_COO!(Ax, Arows, Acols, Avals, x)
    Ax .= zero(eltype(Ax))
    @inbounds @simd for i=1:length(Acols)
        Ax[Arows[i]] += Avals[i] * x[Acols[i]]
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

function get_diag_sparseCOO(Qrows, Qcols, Qvals, n_cols)
    # get diagonal index of M.nzval
    # we assume all columns of M are non empty, and M triangular (:L or :U)
    T = eltype(Qvals)
    n = length(Qrows)
    diagval = zeros(T, n_cols)
    @inbounds @simd for i=1:n
        if Qrows[i] == Qcols[i]
            diagval[Qrows[i]] = Qvals[i]
        end
    end
    return diagval
end

function create_J_augm(IntData, tmp_diag, Qvals, Avals, regu, T)
    if regu.dynamic
        J_augmrows = vcat(IntData.Qcols, IntData.Acols, 1:IntData.n_cols)
        J_augmcols = vcat(IntData.Qrows, IntData.Arows.+IntData.n_cols, 1:IntData.n_cols)
        J_augmvals = vcat(.-Qvals, Avals, tmp_diag)
    else
        J_augmrows = vcat(IntData.Qcols, IntData.Acols, IntData.n_cols+1:IntData.n_cols+IntData.n_rows,
                          1:IntData.n_cols)
        J_augmcols = vcat(IntData.Qrows, IntData.Arows.+IntData.n_cols,
                          IntData.n_cols+1:IntData.n_cols+IntData.n_rows, 1:IntData.n_cols)
        J_augmvals = vcat(.-Qvals, Avals, regu.δ.*ones(T, IntData.n_rows), tmp_diag)
    end
    J_augm = sparse(J_augmrows, J_augmcols, J_augmvals,
                    IntData.n_rows+IntData.n_cols, IntData.n_rows+IntData.n_cols)
    return J_augm
end
