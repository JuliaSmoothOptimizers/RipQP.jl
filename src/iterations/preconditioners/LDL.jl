export LDL

"""
    preconditioner = LDL(; T = Float32, pos = :C, warm_start = true)
Preconditioner for [`K2KrylovParams`](@ref) using a LDL factorization in precision `T`.
The `pos` argument is used to choose the type of preconditioning with an unsymmetric Krylov method.
It can be `:C` (center), `:L` (left) or `:R` (right).
The `warm_start` argument tells RipQP to solve the system with the LDL factorization before using the Krylov method with the LDLFactorization as a preconditioner.
"""
mutable struct LDL{FloatType <: DataType} <: AbstractPreconditioner
  T::FloatType
  pos::Symbol # :L (left), :R (right) or :C (center)
  warm_start::Bool
end

LDL(; T::DataType = Float32, pos = :C, warm_start = true) = LDL(T, pos, warm_start)

mutable struct LDLData{T <: Real, S, Tlow, Op <: Union{LinearOperator, LRPrecond}} <:
               PreconditionerData{T, S}
  K::Symmetric{T, SparseMatrixCSC{T, Int}}
  regu::Regularization{Tlow}
  K_fact::LDLFactorizations.LDLFactorization{Tlow, Int, Int, Int} # factorized matrix
  tmp_res::Vector{Tlow}
  tmp_v::Vector{Tlow}
  fact_fail::Bool # true if factorization failed
  warm_start::Bool
  P::Op
end

precond_name(pdat::LDLData{T, S, Tlow}) where {T, S, Tlow} =
  string(Tlow, " ", string(typeof(pdat).name.name)[1:(end - 4)])

types_linop(op::LinearOperator{T, I, F, Ftu, Faw, S}) where {T, I, F, Ftu, Faw, S} =
  T, I, F, Ftu, Faw, S

lowtype(pdat::LDLData{T, S, Tlow}) where {T, S, Tlow} = Tlow

function ldiv_stor!(
  res,
  K_fact::LDLFactorizations.LDLFactorization{T, Int, Int, Int},
  v,
  tmp_res::Vector{T},
  tmp_v::Vector{T},
) where {T}
  copyto!(tmp_v, v)
  ldiv!(tmp_res, K_fact, tmp_v)
  copyto!(res, tmp_res)
end

function ld_div!(y, b, n, Lp, Li, Lx, D, P)
  y .= b
  z = @views y[P]
  LDLFactorizations.ldl_lsolve!(n, z, Lp, Li, Lx)
end

function dlt_div!(y, b, n, Lp, Li, Lx, D, P)
  y .= b
  z = @views y[P]
  LDLFactorizations.ldl_dsolve!(n, z, D)
  LDLFactorizations.ldl_ltsolve!(n, z, Lp, Li, Lx)
end

function ld_div_stor!(
  res,
  K_fact::LDLFactorizations.LDLFactorization{T, Int, Int, Int},
  v,
  tmp_res::Vector{T},
  tmp_v::Vector{T},
) where {T}
  copyto!(tmp_v, v)
  ld_div!(tmp_res, K_fact, tmp_v)
  copyto!(res, tmp_res)
end

function ld_div_stor!(res, v, tmp_res::Vector{T}, tmp_v::Vector{T}, n, Lp, Li, Lx, D, P) where {T}
  copyto!(tmp_v, v)
  ld_div!(tmp_res, tmp_v, n, Lp, Li, Lx, D, P)
  copyto!(res, tmp_res)
end

function dlt_div_stor!(res, v, tmp_res::Vector{T}, tmp_v::Vector{T}, n, Lp, Li, Lx, D, P) where {T}
  copyto!(tmp_v, v)
  dlt_div!(tmp_res, tmp_v, n, Lp, Li, Lx, D, P)
  copyto!(res, tmp_res)
end

function PreconditionerData(
  sp::AugmentedKrylovParams{<:LDL},
  id::QM_IntData,
  fd::QM_FloatData{T},
  regu::Regularization{T},
  D::AbstractVector{T},
  K,
) where {T <: Real}
  Tlow = sp.preconditioner.T
  @assert fd.uplo == :U
  sp.form_mat = true
  regu_precond = Regularization(
    -Tlow(D[1]),
    Tlow(max(regu.δ, sqrt(eps(Tlow)))),
    sqrt(eps(Tlow)),
    sqrt(eps(Tlow)),
    regu.δ != 0 ? :classic : :dynamic,
  )
  K_fact = @timeit_debug to "LDL analyze" ldl_analyze(K, Tf = Tlow)

  return PreconditionerData(sp, K_fact, id.nvar, id.ncon, regu_precond, K)
end

function PreconditionerData(
  sp::AugmentedKrylovParams{<:LDL},
  K_fact::LDLFactorizations.LDLFactorization{Tlow},
  nvar::Int,
  ncon::Int,
  regu_precond::Regularization{Tlow},
  K,
) where {Tlow <: Real}
  regu_precond.regul = :dynamic
  T = eltype(K)
  if regu_precond.regul == :dynamic
    regu_precond.ρ, regu_precond.δ = -Tlow(eps(Tlow)^(3 / 4)), Tlow(eps(Tlow)^(0.45))
    K_fact.r1, K_fact.r2 = regu_precond.ρ, regu_precond.δ
    K_fact.tol = Tlow(eps(Tlow))
    K_fact.n_d = nvar
  end

  if T == Tlow
    if sp.kmethod == :gmres || sp.kmethod == :dqgmres
      if sp.preconditioner.pos == :C
        M = LinearOperator(
          T,
          nvar + ncon,
          nvar + ncon,
          false,
          false,
          (res, v) -> ld_div!(res, v, K_fact.n, K_fact.Lp, K_fact.Li, K_fact.Lx, K_fact.d, K_fact.P),
        )
        N = LinearOperator(
          T,
          nvar + ncon,
          nvar + ncon,
          false,
          false,
          (res, v) -> dlt_div!(res, v, K_fact.n, K_fact.Lp, K_fact.Li, K_fact.Lx, K_fact.d, K_fact.P),
        )
      elseif sp.preconditioner.pos == :L
        M = LinearOperator(
          T,
          nvar + ncon,
          nvar + ncon,
          true,
          true,
          (res, v) -> ldiv!(res, K_fact, v),
        )
        N = I
      elseif sp.preconditioner.pos == :R
        M = I
        N = LinearOperator(
          T,
          nvar + ncon,
          nvar + ncon,
          true,
          true,
          (res, v) -> ldiv!(res, K_fact, v),
        )
      end
      P = LRPrecond(M, N)
    else
      K_fact.d .= abs.(K_fact.d)
      P = LinearOperator(
        Tlow,
        nvar + ncon,
        nvar + ncon,
        true,
        true,
        (res, v, α, β) -> ldiv!(res, K_fact, v),
      )
    end
  else
    tmp_res = Vector{Tlow}(undef, nvar + ncon)
    tmp_v = Vector{Tlow}(undef, nvar + ncon)
    if sp.kmethod == :gmres || sp.kmethod == :dqgmres
      if sp.preconditioner.pos == :C
        M = LinearOperator(
          T,
          nvar + ncon,
          nvar + ncon,
          false,
          false,
          (res, v) -> ld_div_stor!(res, v, tmp_res, tmp_v, K_fact.n, K_fact.Lp, K_fact.Li, K_fact.Lx, K_fact.d, K_fact.P),
        )
        N = LinearOperator(
          T,
          nvar + ncon,
          nvar + ncon,
          false,
          false,
          (res, v) -> dlt_div_stor!(res, v, tmp_res, tmp_v, K_fact.n, K_fact.Lp, K_fact.Li, K_fact.Lx, K_fact.d, K_fact.P),
        )
      elseif sp.preconditioner.pos == :L
        M = LinearOperator(
          T,
          nvar + ncon,
          nvar + ncon,
          true,
          true,
          (res, v) -> ldiv_stor!(res, K_fact, v, tmp_res, tmp_v),
        )
        N = I
      elseif sp.preconditioner.pos == :R
        M = I
        N = LinearOperator(
          T,
          nvar + ncon,
          nvar + ncon,
          true,
          true,
          (res, v) -> ldiv_stor!(res, K_fact, v, tmp_res, tmp_v),
        )
      end
      P = LRPrecond(M, N)
    else
      K_fact.d .= abs.(K_fact.d)
      P = LinearOperator(
        T,
        nvar + ncon,
        nvar + ncon,
        true,
        true,
        (res, v, α, β) -> ldiv_stor!(res, K_fact, v, tmp_res, tmp_v),
      )
    end
  end
  return LDLData{T, Vector{T}, Tlow, typeof(P)}(
    K,
    regu_precond,
    K_fact,
    T == Tlow ? T[] : tmp_res,
    T == Tlow ? T[] : tmp_v,
    false,
    sp.preconditioner.warm_start,
    P,
  )
end

function factorize_scale_K2!(
  K,
  K_fact,
  D,
  Deq,
  diag_Q,
  diagind_K,
  regu,
  s_l,
  s_u,
  x_m_lvar,
  uvar_m_x,
  ilow,
  iupp,
  ncon,
  nvar,
  cnts,
  qp,
  T,
  T0,
)
  if regu.regul == :dynamic
    update_K_dynamic!(K, K_fact, regu, diagind_K, cnts, T, qp)
    @timeit_debug to "LDL factorize" ldl_factorize!(Symmetric(K, :U), K_fact)
  elseif regu.regul == :classic
    @timeit_debug to "LDL factorize" ldl_factorize!(Symmetric(K, :U), K_fact)
    while !factorized(K_fact)
      out = update_regu_trycatch!(regu, cnts, T, T0)
      out == 1 && return out
      cnts.c_catch += 1
      cnts.c_catch >= 4 && return 1
      update_K!(
        K,
        D,
        regu,
        s_l,
        s_u,
        x_m_lvar,
        uvar_m_x,
        ilow,
        iupp,
        diag_Q,
        diagind_K,
        nvar,
        ncon,
        T,
      )
      @timeit_debug to "LDL factorize" ldl_factorize!(Symmetric(K, :U), K_fact)
    end

  else # no Regularization
    @timeit_debug to "LDL factorize" ldl_factorize!(Symmetric(K, :U), K_fact)
  end

  return 0 # factorization succeeded
end

function update_preconditioner!(
  pdat::LDLData{T},
  pad::PreallocatedData{T},
  itd::IterData{T},
  pt::Point{T},
  id::QM_IntData,
  fd::Abstract_QM_FloatData{T},
  cnts::Counters,
) where {T <: Real}
  Tlow = lowtype(pad.pdat)
  pad.pdat.regu.ρ, pad.pdat.regu.δ =
    max(pad.regu.ρ, sqrt(eps(Tlow))), max(pad.regu.ρ, sqrt(eps(Tlow)))

  out = factorize_scale_K2!(
    pad.pdat.K.data,
    pad.pdat.K_fact,
    pad.D,
    pad.mt.Deq,
    pad.mt.diag_Q,
    pad.mt.diagind_K,
    pad.pdat.regu,
    pt.s_l,
    pt.s_u,
    itd.x_m_lvar,
    itd.uvar_m_x,
    id.ilow,
    id.iupp,
    id.ncon,
    id.nvar,
    cnts,
    itd.qp,
    Tlow,
    Tlow,
  ) # update D and factorize K

  if out == 1
    pad.pdat.fact_fail = true
    return out
  end
  if pad.pdat.warm_start
    ldiv!(pad.KS.x, pad.pdat.K_fact, pad.rhs)
    warm_start!(pad.KS, pad.KS.x)
  end
  if !(typeof(pad.KS) <: GmresSolver || typeof(pad.KS) <: DqgmresSolver)
    pad.pdat.K_fact.d .= abs.(pad.pdat.K_fact.d)
  end
end
