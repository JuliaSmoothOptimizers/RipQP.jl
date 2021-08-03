# shift ivec so that its coeffs corresponds to the new lvar and uvar after removing fixed variables
function shift_ivector!(ivec, ifix)
  nvec = length(ivec)
  nfix = length(ifix)
  if nvec > 0
    cfix = 1
    c_vec = 1
    for i = 1:nvec
      while cfix <= nfix && ivec[i] > ifix[cfix]
        cfix += 1
      end
      ivec[i] -= cfix - 1
    end
  end
end

function reverseshift_ivector!(ivec, ifix)
  nvec = length(ivec)
  nfix = length(ifix)
  if nvec > 0
    cfix = 1
    for i = 1:nvec
      ivec[i] += cfix - 1
      while cfix <= nfix && ivec[i] >= ifix[cfix]
        cfix += 1
        ivec[i] += 1
      end
    end
  end
end

# ̃xᵀ̃Q̃x̃ + ̃ĉᵀx̃ + lⱼ²Qⱼⱼ + cⱼxⱼ + c₀
# ̂c = ̃c + 2lⱼΣₖQⱼₖxₖ , k ≂̸ j  
function remove_ifix!(
  ifix,
  Qcolptr,
  Qrowval,
  Qnzval,
  Qn,
  Acolptr,
  Arowval,
  Anzval,
  An,
  c::AbstractVector{T},
  c0,
  lvar,
  uvar,
  lcon,
  ucon,
  ilow,
  iupp,
  irng,
  ifree,
  uplo,
) where {T}
  # assume ifix is sorted and length(ifix) > 0
  # assume Qcols is sorted
  c0_offset = zero(T)
  Qnnz = length(Qrowval)
  Annz = length(Arowval)
  nfix = length(ifix)
  for idxfix = 1:nfix
    currentifix = ifix[idxfix]
    xifix = lvar[currentifix]
    newcurrentifix = currentifix - idxfix + 1
    Qwritepos = 1
    oldQcolptrQj = 1
    shiftQj = 1 # increase Qj of currentj - 1 if Qj
    # remove ifix in Q and update data
    for Qj = 1:(Qn - idxfix + 1)
      while shiftQj <= idxfix - 1 && Qj + shiftQj - 1 >= ifix[shiftQj]
        shiftQj += 1
      end
      shiftQi = 1
      for Qk = oldQcolptrQj:(Qcolptr[Qj + 1] - 1)
        Qi, Qx = Qrowval[Qk], Qnzval[Qk]
        while shiftQi <= idxfix - 1 && Qi + shiftQi - 1 >= ifix[shiftQi]
          shiftQi += 1
        end
        if Qi == Qj == newcurrentifix
          c0_offset += xifix^2 * Qx / 2
        elseif Qi == newcurrentifix
          c[Qj + shiftQj - 1] += xifix * Qx
        elseif Qj == newcurrentifix
          c[Qi + shiftQi - 1] += xifix * Qx
        else
          Qrowval[Qwritepos] = (Qi < newcurrentifix) ? Qi : Qi - 1
          Qnzval[Qwritepos] = Qx
          Qwritepos += 1
        end
      end
      oldQcolptrQj = Qcolptr[Qj + 1]
      if Qj >= newcurrentifix
        Qcolptr[Qj] = Qwritepos
      else
        Qcolptr[Qj + 1] = Qwritepos
      end
    end

    # remove ifix in A cols
    Awritepos = 1
    oldAcolptrAj = 1
    currentAn = (uplo == :L) ? An - idxfix + 1 : An # remove rows if uplo == :U 
    for Aj = 1:currentAn
      for Ak = oldAcolptrAj:(Acolptr[Aj + 1] - 1)
        Ai, Ax = Arowval[Ak], Anzval[Ak]
        if uplo == :L
          if Aj == newcurrentifix
            lcon[Ai] -= Ax * xifix
            ucon[Ai] -= Ax * xifix
          else
            if Awritepos != Ak
              Arowval[Awritepos] = Ai
              Anzval[Awritepos] = Ax
            end
            Awritepos += 1
          end
        elseif uplo == :U # A is actually Aᵀ in this case
          if Ai == newcurrentifix
            lcon[Aj] -= Ax * xifix
            ucon[Aj] -= Ax * xifix
          else
            Arowval[Awritepos] = (Ai < newcurrentifix) ? Ai : Ai - 1
            Anzval[Awritepos] = Ax
            Awritepos += 1
          end
        end
      end
      oldAcolptrAj = Acolptr[Aj + 1]
      if Aj >= newcurrentifix && uplo == :L
        Acolptr[Aj] = Awritepos
      else
        Acolptr[Aj + 1] = Awritepos
      end
    end

    # update c0 with c[currentifix] coeff
    c0_offset += c[currentifix] * xifix
  end

  # resize Q and A
  Qnnz = Qcolptr[end - nfix] - 1
  Annz = (uplo == :L) ? Acolptr[end - nfix] - 1 : Acolptr[end] - 1
  resize!(Qcolptr, Qn + 1 - nfix)
  resize!(Qrowval, Qnnz)
  resize!(Qnzval, Qnnz)
  (uplo == :L) && resize!(Acolptr, An + 1 - nfix)
  resize!(Arowval, Annz)
  resize!(Anzval, Annz)

  # store removed x values
  xrm = lvar[ifix]

  # remove coefs in lvar, uvar, c
  deleteat!(c, ifix)
  deleteat!(lvar, ifix)
  deleteat!(uvar, ifix)

  # shift ilow, iupp, irng, ifree, iinf
  shift_ivector!(ilow, ifix)
  shift_ivector!(iupp, ifix)
  shift_ivector!(irng, ifix)
  shift_ivector!(ifree, ifix)

  # update c0
  c0 += c0_offset

  nvarrm = Qn - nfix

  return xrm, c0, nvarrm, lvar, uvar, lcon, ucon
end

function restore_ifix!(ifix, ilow, iupp, irng, ifree, xrm, x, xout)
  reverseshift_ivector!(ilow, ifix)
  reverseshift_ivector!(iupp, ifix)
  reverseshift_ivector!(irng, ifix)
  reverseshift_ivector!(ifree, ifix)

  # put x and xrm inside xout
  cfix, cx = 1, 1
  nfix = length(ifix)
  for i = 1:length(xout)
    if cfix <= nfix && i == ifix[cfix]
      xout[i] = xrm[cfix]
      cfix += 1
    else
      xout[i] = x[cx]
      cx += 1
    end
  end
end

function rm_rowcolQ(Q, ifix)
  Qcolptr = copy(Q.colptr)
  Qrowval = copy(Q.rowval)
  Qnzval = copy(Q.nzval)
  Qn = size(Q, 2)

  nfix = length(ifix)
  nbrmQ = 0
  # remove ifix in Q and update data
  for idxfix = 1:nfix
    Qwritepos = 1
    oldQcolptrQj = 1
    currentifix = ifix[idxfix] - idxfix + 1
    for Qj = 1:(Qn - idxfix + 1)
      for Qk = oldQcolptrQj:(Qcolptr[Qj + 1] - 1)
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
      oldQcolptrQj = Qcolptr[Qj + 1]
      if Qj >= currentifix
        Qcolptr[Qj] = Qwritepos
      else
        Qcolptr[Qj + 1] = Qwritepos
      end
    end
  end
  Qnnz = Qcolptr[end - nfix] - 1
  resize!(Qrowval, Qnnz)
  resize!(Qnzval, Qnnz)
  resize!(Qcolptr, Qn + 1 - nfix)
  return SparseMatrixCSC(Qn - nfix, Qn - nfix, Qcolptr, Qrowval, Qnzval)
end
