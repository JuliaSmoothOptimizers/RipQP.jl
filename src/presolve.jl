# shift ivec so that its coeffs corresponds to the new lvar and uvar after removing fixed variables
function shift_ivector!(ivec, ifix)
  nvec = length(ivec)
  nfix = length(ifix)
  if nvec > 0
    c_fix = 1
    c_vec = 1
    for i= 1:nvec
      while c_fix <= nfix && ivec[i] > ifix[c_fix]
        c_fix += 1
      end
      ivec[i] -= c_fix - 1
    end
  end
end

function remove_ifix!(ifix, Qcolptr, Qrowval, Qnzval, Qn, Acolptr, Arowval, Anzval, An,
                      c::AbstractVector{T}, c0, lvar, uvar, lcon, ucon,
                      ilow, iupp, irng, ifree, iinf) where T
  # assume ifix is sorted and length(ifix) > 0
  # assume Qcols is sorted
  c0_offset = zero(T)
  Qnnz = length(Qrowval)
  Annz = length(Arowval)
  nbrmQ = 0
  nbrmA = 0
  cfix = 1
  for currentifix in ifix
    xifix = lvar[currentifix]
    Qwritepos = 1
    oldQcolptrQj = 1
    # remove ifix in Q and update data
    for Qj in 1:Qn
      for Qk in oldQcolptrQj:(Qcolptr[Qj+1] - 1)
        Qi, Qx = Qrowval[Qk], Qnzval[Qk]
        if Qrowi == Qcoli == currentifix
          c0_offset += xifix^2 * Qvali
          nbrmQ += 1
        elseif Qrowi == currentifix
          c[Qcoli] += 2 * xifix * Qvali
          nbrmQ += 1
        elseif Qcoli == currentifix
          c[Qvali] += 2 * xifix * Qvali
          nbrmQ += 1
        else
          if Qwritepos != Qk
            Qrowval[Qwritepos] = Qi
            Qnzval[Qwritepos] = Qx
            Qvals[Qwritepos] = Qvali
          end
          Qwritepos += 1
        end
      end
      if Qj != currentifix
        oldQcolptrQj = Qcolptr[Qj+1]
        Qcolptr[Qj+1] = Qwritepos
      end
    end
    # remove ifix in A cols
    for i = 1:nnzA
      Arowi, Acoli, Avali = Arows[i], Acols[i], Avals[i]
      if Acoli == currentifix
        lcon[Arowi] -= Avali * xifix
        ucon[Arowi] -= Avali * xifix
        nbrmA += 1
      else
        Awritepos += 1
        Arows[Awritepos] = Arowi
        Acols[Awritepos] = Acoli
        Avals[Awritepos] = Avali
      end
    end
    # update c0 with c[currentifix] coeff
    c0_offset += c[currentifix] * xifix
  end
  # resize Q and A
  nnzQnew, nnzAnew = nnzQ - nbrmQ, nnzA - nbrmA
  resize!(Qrows, nnzQnew)
  resize!(Qcols, nnzQnew)
  resize!(Qvals, nnzQnew)
  resize!(Arows, nnzAnew)
  resize!(Acols, nnzAnew)
  resize!(Avals, nnzAnew)
  # remove coefs in lvar, uvar, c
  deleteat!(c, ifix)
  deleteat!(lvar, ifix)
  deleteat!(uvar, ifix)
  # shift ilow, iupp, irng, ifree, iinf
  shift_ivector!(ilow, ifix)
  shift_ivector!(iupp, ifix)
  shift_ivector!(irng, ifix)
  shift_ivector!(ifree, ifix)
  shift_ivector!(iinf, ifix)
  # update c0
  c0 += c0_offset
end

function rm_rowcolQ(Q, ifix)
  Qcolptr = copy(Q.colptr)
  Qrowval = copy(Q.rowval)
  Qnzval = copy(Q.nzval)
  Qn = size(Q, 2)

  currentifix = ifix
  nbrmQ = 0
  Qwritepos = 1
  oldQcolptrQj = 1
  # remove ifix in Q and update data
  for Qj in 1:Qn
    for Qk in oldQcolptrQj:(Qcolptr[Qj+1] - 1)
      Qi, Qx = Qrowval[Qk], Qnzval[Qk]
      if Qi == Qj == currentifix
        nbrmQ += 1
      elseif Qi == currentifix
        nbrmQ += 1
      elseif Qj == currentifix
        nbrmQ += 1
      else
        if Qwritepos != Qk
          Qrowval[Qwritepos] = (Qi < currentifix) ? Qi : Qi - 1
          Qnzval[Qwritepos] = Qx
        end
        Qwritepos += 1
      end
    end
    if Qj != currentifix
      oldQcolptrQj = Qcolptr[Qj+1]
      Qcolptr[Qj+1] = Qwritepos
    end
  end
  Qnnz = Qcolptr[end-1] - 1
  resize!(Qrowval, Qnnz)
  resize!(Qnzval, Qnnz)
  resize!(Qcolptr, Qn)
  return SparseMatrixCSC(Qn-1, Qn-1, Qcolptr, Qrowval, Qnzval)
end