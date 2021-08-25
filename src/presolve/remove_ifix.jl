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
  Qrows,
  Qcols,
  Qvals,
  Qn,
  Arows,
  Acols,
  Avals,
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
  Qnnz = length(Qrows)
  Qrm = 0
  Annz = length(Arows)
  Arm = 0
  nfix = length(ifix)
  for idxfix = 1:nfix
    currentifix = ifix[idxfix]
    xifix = lvar[currentifix]
    newcurrentifix = currentifix - idxfix + 1
    Qwritepos = 1
    oldQcolptrQj = 1
    shiftQj = 1 # increase Qj of currentj - 1 if Qj
    if Qnnz > 0
      oldQj = Qrows[1]
    end
    # remove ifix in Q and update data
    k = 1
    while k <= Qnnz && Qcols[k] <= (Qn - idxfix + 1)
      Qi, Qj, Qx = Qrows[k], Qcols[k], Qvals[k] # Qj sorted 

      while (Qj == oldQj) && shiftQj <= idxfix - 1 && Qj + shiftQj - 1 >= ifix[shiftQj]
        shiftQj += 1
      end
      shiftQi = 1
      while shiftQi <= idxfix - 1 && Qi + shiftQi - 1 >= ifix[shiftQi]
        shiftQi += 1
      end
      if Qi == Qj == newcurrentifix
        Qrm += 1
        c0_offset += xifix^2 * Qx / 2
      elseif Qi == newcurrentifix
        Qrm += 1
        c[Qj + shiftQj - 1] += xifix * Qx
      elseif Qj == newcurrentifix
        Qrm += 1
        c[Qi + shiftQi - 1] += xifix * Qx
      else
        Qrows[Qwritepos] = (Qi < newcurrentifix) ? Qi : Qi - 1
        Qcols[Qwritepos] = (Qj < newcurrentifix) ? Qj : Qj - 1
        Qvals[Qwritepos] = Qx
        Qwritepos += 1
      end
      k += 1
    end

    # remove ifix in A cols
    Awritepos = 1
    oldAcolptrAj = 1
    currentAn = An - idxfix + 1  # remove rows if uplo == :U 
    k = 1
    while k <= Annz && Acols[k] <= currentAn
      Ai, Aj, Ax = Arows[k], Acols[k], Avals[k]
      if Aj == newcurrentifix
        Arm += 1
        lcon[Ai] -= Ax * xifix
        ucon[Ai] -= Ax * xifix
      else
        if Awritepos != k
          Arows[Awritepos] = Ai
          Acols[Awritepos] = (Aj < newcurrentifix) ? Aj : Aj - 1
          Avals[Awritepos] = Ax
        end
        Awritepos += 1
      end
      k += 1
    end

    # update c0 with c[currentifix] coeff
    c0_offset += c[currentifix] * xifix
  end

  # resize Q and A
  if nfix > 0
    Qnnz -= Qrm
    Annz -= Arm
    resize!(Qrows, Qnnz)
    resize!(Qcols, Qnnz)
    resize!(Qvals, Qnnz)
    resize!(Arows, Annz)
    resize!(Acols, Annz)
    resize!(Avals, Annz)
  end

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
