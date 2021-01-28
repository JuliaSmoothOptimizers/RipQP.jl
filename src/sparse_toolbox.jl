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
