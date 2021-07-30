include("sparse_coords.jl")
include("remove_ifix.jl")

function presolve(QM; uplo=:L)
  if uplo == :L
    Qm, Qn, Qcolptr, Qrowval, Qnzval = sparse_coords(QM.data.Hrows, QM.data.Hcols, QM.data.Hvals, QM.meta.nvar, QM.meta.nvar)
    Am, An, Acolptr, Arowval, Anzval = sparse_coords(QM.data.Arows, QM.data.Acols, QM.data.Avals, QM.meta.ncon, QM.meta.nvar)
  else
    Qm, Qn, Qcolptr, Qrowval, Qnzval = sparse_coords(QM.data.Hcols, QM.data.Hrows, QM.data.Hvals, QM.meta.nvar, QM.meta.nvar)
    Am, An, Acolptr, Arowval, Anzval = sparse_coords(QM.data.Acols, QM.data.Arows, QM.data.Avals, QM.meta.nvar, QM.meta.ncon)
  end
  lvar, uvar, lcon, ucon = QM.meta.lvar, QM.meta.uvar, QM.meta.lcon, QM.meta.ucon
  c, c0 = QM.data.c, QM.data.c0
  ilow, iupp, irng, ifree = QM.meta.ilow, QM.meta.iupp, QM.meta.irng, QM.meta.ifree
  ifix = QM.meta.ifix
  ncon = QM.meta.ncon

  if length(ifix) > 0
    xrm, c0, nvarrm = remove_ifix!(ifix, Qcolptr, Qrowval, Qnzval, Qn, Acolptr, Arowval, Anzval, An, c, c0, lvar, uvar, lcon, ucon,
                          ilow, iupp, irng, ifree, uplo)
  end

  Q = SparseMatrixCSC(nvarrm, nvarrm, Qcolptr, Qrowval, Qnzval)
  if uplo == :L
    A = SparseMatrixCSC(ncon, nvarrm, Acolptr, Arowval, Anzval)
  else
    A = SparseMatrixCSC(nvarrm, ncon, Acolptr, Arowval, Anzval)
  end
  return Q, A, xrm
end

function postsolve!(fd::QM_FloatData{T}, id::QM_IntData, pt::Point{T}, ps::PresolveData{T}) where {T <: Real}
  if length(ifix) > 0
    restore_ifix!(id.ifix, id.ilow, id.iupp, id.irng, id.ifree, ps.xrm, pt.x, ps.xout)
    pt.x = ps.xout
  end
end

# compare true 
function rmfix2(QM)
  Qrows, Qcols, Qvals = copy(QM.data.Hrows), copy(QM.data.Hcols), copy(QM.data.Hvals)
  Arows, Acols, Avals = copy(QM.data.Arows), copy(QM.data.Acols), copy(QM.data.Avals)
  Qrows, Qcols, Qvals, c, c0, Arows, Acols, Avals, Lcon, Ucon,
            lvar, uvar, n_cols, Arows_rm_fix, Acols_rm_fix, Avals_rm_fix,
            Qrows_rm_fix, Qcols_rm_fix, Qvals_rm_fix, c_rm_fix, x_rm_fix, ifix = 
            rm_ifix2!(copy(QM.meta.ifix), Qrows, Qcols, Qvals, QM.data.c, QM.data.c0, Arows, Acols, Avals,
                  qm.meta.lcon, QM.meta.ucon, qm.meta.lvar, QM.meta.uvar, QM.meta.ncon, QM.meta.nvar)

  Q = sparse(Qrows, Qcols, Qvals, n_cols, n_cols)
  A = sparse(Arows, Acols, Avals, QM.meta.ncon, n_cols)
  return Q, A, x_rm_fix
end

function rm_ifix2!(ifix, Qrows, Qcols, Qvals, c, c0, Arows, Acols, Avals,
                  Lcon, Ucon, lvar, uvar, n_rows, n_cols)
    T = eltype(c)
    # get Qii
    diag_Q = zeros(T, n_cols)
    for i=1:length(Qvals)
        if Qrows[i] == Qcols[i]
            diag_Q[Qrows[i]] = Qvals[i]
        end
    end
    ifix = sort!(ifix)
    # update c0, c
    Qji = zero(T)
    for i=1:length(ifix)
        c0 += c[ifix[i]] * lvar[ifix[i]] + diag_Q[ifix[i]] * lvar[ifix[i]]^2 / 2
        for j=1:n_cols
            Qji = zero(T)
            for k=1:length(Qvals)
                if (Qrows[k] == i && Qcols[k] == j) || (Qrows[k] == j && Qcols[k] == i)
                    Qji += Qvals[k]
                end
            end
            c[j] += lvar[ifix[i]] * Qji
        end
    end

    # remove columns in ifix
    ifix_cols_A = findall(x->x in ifix, Acols)
    ifix_cols_A = sort!(ifix_cols_A)
    for i=1:length(Acols)
        if i in ifix_cols_A
            Lcon[Arows[i]] -= Avals[i] * lvar[Acols[i]]
            Ucon[Arows[i]] -= Avals[i] * lvar[Acols[i]]
        end
    end
    Arows_rm_fix = Arows[ifix_cols_A]
    Acols_rm_fix = Acols[ifix_cols_A]
    Avals_rm_fix = Avals[ifix_cols_A]
    Arows = deleteat!(Arows, ifix_cols_A)
    Acols = deleteat!(Acols, ifix_cols_A)
    Avals = deleteat!(Avals, ifix_cols_A)

    for i=1:length(Acols)
        if Acols[i] > ifix[1]
            Acols[i] -= findlast(ifix .<= Acols[i])
        end
    end
    # remove rows and columns in ifix
    ifix_cols_Q = findall(x-> x in ifix, Qcols)

    Q_rm_idx = [] #unique(hcat(ifix_rows_Q, ifix_cols_Q))
    Qrows_rm_fix = Qrows[Q_rm_idx]
    Qcols_rm_fix = Qcols[Q_rm_idx]
    Qvals_rm_fix = Qvals[Q_rm_idx]
    for i=1:length(ifix_cols_Q)
        Qrows_rm_fix = push!(Qrows_rm_fix, splice!(Qrows, ifix_cols_Q[i]-i+1))
        Qcols_rm_fix = push!(Qcols_rm_fix, splice!(Qcols, ifix_cols_Q[i]-i+1))
        Qvals_rm_fix = push!(Qvals_rm_fix, splice!(Qvals, ifix_cols_Q[i]-i+1))
    end
    ifix_rows_Q = findall(x-> x in ifix, Qrows)
    for i=1:length(ifix_rows_Q)
        Qrows_rm_fix = push!(Qrows_rm_fix, splice!(Qrows, ifix_rows_Q[i]-i+1))
        Qcols_rm_fix = push!(Qcols_rm_fix, splice!(Qcols, ifix_rows_Q[i]-i+1))
        Qvals_rm_fix = push!(Qvals_rm_fix, splice!(Qvals, ifix_rows_Q[i]-i+1))
    end

    for i=1:length(Qcols)
        if  Qrows[i] > ifix[1]
            Qrows[i] -= findlast(ifix .<= Qrows[i])
        end
        if Qcols[i] > ifix[1]
            Qcols[i] -= findlast(ifix .<= Qcols[i])
        end
    end

    c_rm_fix = c[ifix]
    x_rm_fix = lvar[ifix]
    c = deleteat!(c, ifix)
    lvar = deleteat!(lvar, ifix)
    uvar = deleteat!(uvar, ifix)
    n_cols -= length(ifix)

    return Qrows, Qcols, Qvals, c, c0, Arows, Acols, Avals, Lcon, Ucon,
            lvar, uvar, n_cols, Arows_rm_fix, Acols_rm_fix, Avals_rm_fix,
            Qrows_rm_fix, Qcols_rm_fix, Qvals_rm_fix, c_rm_fix, x_rm_fix, ifix
end