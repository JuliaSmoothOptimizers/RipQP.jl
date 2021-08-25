include("sparse_coords.jl")
include("remove_ifix.jl")

function presolveQM(QM::QuadraticModel{T, S}; uplo = :L) where {T <: Real, S}

  Qrows, Qcols, Qvals, Qm, Qn = QM.data.Hrows, QM.data.Hcols, QM.data.Hvals, QM.meta.nvar, QM.meta.nvar
  Arows, Acols, Avals, Am, An = QM.data.Arows, QM.data.Acols, QM.data.Avals, QM.meta.ncon, QM.meta.nvar
  lvar, uvar, lcon, ucon = QM.meta.lvar, QM.meta.uvar, QM.meta.lcon, QM.meta.ucon
  c, c0 = QM.data.c, QM.data.c0
  ilow, iupp, irng, ifree = QM.meta.ilow, QM.meta.iupp, QM.meta.irng, QM.meta.ifree
  ifix = QM.meta.ifix
  ncon = QM.meta.ncon

  if length(ifix) > 0
    xrm, c0, nvarrm, lvar, uvar, lcon, ucon = remove_ifix!(
      ifix,
      Qrows,
      Qcols,
      Qvals,
      Qn,
      Arows,
      Acols,
      Avals,
      An,
      c,
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
    )
  else
    nvarrm = QM.meta.nvar
    xrm = S(undef, 0)
  end

  # return Acolptr, Arowval, Anzval
  if uplo == :L
    Q = sparse(Qrows, Qcols, Qvals, nvarrm, nvarrm)
    A = sparse(Arows, Acols, Avals, ncon, nvarrm)
  else
    Q = sparse(Qcols, Qrows, Qvals, nvarrm, nvarrm)
    A = sparse(Acols, Arows, Avals, nvarrm, ncon)
  end
  return Q, A, xrm, c0, nvarrm, lvar, uvar, lcon, ucon, ilow, iupp, irng, ifree, ifix
end

function postsolve!(fd, id, pt, ps)
  if length(id.ifix) > 0
    restore_ifix!(id.ifix, id.ilow, id.iupp, id.irng, id.ifree, ps.xrm, pt.x, ps.xout)
    pt.x = ps.xout
  end
end
