export K2KrylovParams

"""
Type to use the K2 formulation with a Krylov method, using the package 
[`Krylov.jl`](https://github.com/JuliaSmoothOptimizers/Krylov.jl). 
The outer constructor 

    K2KrylovParams(; uplo = :L, kmethod = :minres, preconditioner = Identity(),
                   rhs_scale = true, form_mat = false, equilibrate = false,
                   atol0 = 1.0e-4, rtol0 = 1.0e-4, 
                   atol_min = 1.0e-10, rtol_min = 1.0e-10,
                   ρ0 = sqrt(eps()) * 1e5, δ0 = sqrt(eps()) * 1e5, 
                   ρ_min = 1e2 * sqrt(eps()), δ_min = 1e2 * sqrt(eps()),
                   itmax = 0, memory = 20)

creates a [`RipQP.SolverParams`](@ref).
"""
mutable struct K2KrylovParams{T, PT, FT <: DataType} <: AugmentedKrylovParams{T, PT}
  uplo::Symbol
  kmethod::Symbol
  preconditioner::PT
  rhs_scale::Bool
  form_mat::Bool
  equilibrate::Bool
  atol0::T
  rtol0::T
  atol_min::T
  rtol_min::T
  ρ0::T
  δ0::T
  ρ_min::T
  δ_min::T
  itmax::Int
  mem::Int
  Tir::FT
end

function K2KrylovParams{T}(;
  uplo::Symbol = :L,
  kmethod::Symbol = :minres,
  preconditioner::AbstractPreconditioner = Identity(),
  rhs_scale::Bool = true,
  form_mat::Bool = false,
  equilibrate::Bool = false,
  atol0::T = T(eps(T)^(1 / 4)),
  rtol0::T = T(eps(T)^(1 / 4)),
  atol_min::T = sqrt(eps(T)),
  rtol_min::T = sqrt(eps(T)),
  ρ0::T = T(sqrt(eps()) * 1e5),
  δ0::T = T(sqrt(eps()) * 1e5),
  ρ_min::T = T(1e2 * sqrt(eps(T))),
  δ_min::T = T(1e2 * sqrt(eps(T))),
  itmax::Int = 0,
  mem::Int = 20,
  Tir::DataType = T,
) where {T <: Real}
  if equilibrate && !form_mat
    error("use form_mat = true to use equilibration")
  end
  if typeof(preconditioner) <: LDL
    if !form_mat
      form_mat = true
      @info "changed form_mat to true to use this preconditioner"
    end
    uplo_fact = get_uplo(preconditioner.fact_alg)
    if uplo != uplo_fact
      uplo = uplo_fact
      @info "changed uplo to :$uplo_fact to use this preconditioner"
    end
  end
  return K2KrylovParams(
    uplo,
    kmethod,
    preconditioner,
    rhs_scale,
    form_mat,
    equilibrate,
    atol0,
    rtol0,
    atol_min,
    rtol_min,
    ρ0,
    δ0,
    ρ_min,
    δ_min,
    itmax,
    mem,
    Tir,
  )
end

K2KrylovParams(; kwargs...) = K2KrylovParams{Float64}(; kwargs...)

mutable struct MatrixTools{T}
  diag_Q::SparseVector{T, Int} # Q diag
  diagind_K::Vector{Int}
  Deq::Diagonal{T, Vector{T}}
  C_eq::Diagonal{T, Vector{T}}
end

convert(::Type{MatrixTools{T}}, mt::MatrixTools) where {T} = MatrixTools(
  convert(SparseVector{T, Int}, mt.diag_Q),
  mt.diagind_K,
  Diagonal(convert(Vector{T}, mt.Deq.diag)),
  Diagonal(convert(Vector{T}, mt.C_eq.diag)),
)

mutable struct PreallocatedDataK2Krylov{
  T <: Real,
  S,
  M <: Union{LinearOperator{T}, AbstractMatrix{T}},
  MT <: Union{MatrixTools{T}, Int},
  Pr <: PreconditionerData,
  Ksol <: KrylovSolver,
} <: PreallocatedDataAugmentedKrylov{T, S}
  pdat::Pr
  D::S                                  # temporary top-left diagonal
  rhs::S
  rhs_scale::Bool
  equilibrate::Bool
  regu::Regularization{T}
  δv::Vector{T}
  K::M # augmented matrix
  mt::MT
  KS::Ksol
  kiter::Int
  atol::T
  rtol::T
  atol_min::T
  rtol_min::T
  itmax::Int
end

function opK2prod!(
  res::AbstractVector{T},
  nvar::Int,
  Q::Union{AbstractMatrix{T}, AbstractLinearOperator{T}},
  D::AbstractVector{T},
  A::Union{AbstractMatrix{T}, AbstractLinearOperator{T}},
  δv::AbstractVector{T},
  v::AbstractVector{T},
  α::T,
  β::T,
  uplo::Symbol,
) where {T}
  @views mul!(res[1:nvar], Q, v[1:nvar], -α, β)
  res[1:nvar] .+= α .* D .* v[1:nvar]
  if uplo == :U
    @views mul!(res[1:nvar], A, v[(nvar + 1):end], α, one(T))
    @views mul!(res[(nvar + 1):end], A', v[1:nvar], α, β)
  else
    @views mul!(res[1:nvar], A', v[(nvar + 1):end], α, one(T))
    @views mul!(res[(nvar + 1):end], A, v[1:nvar], α, β)
  end
  res[(nvar + 1):end] .+= @views (α * δv[1]) .* v[(nvar + 1):end]
end

function PreallocatedData(
  sp::K2KrylovParams,
  fd::QM_FloatData{T},
  id::QM_IntData,
  itd::IterData{T},
  pt::Point{T},
  iconf::InputConfig{Tconf},
) where {T <: Real, Tconf <: Real}

  # init Regularization values
  D = similar(fd.c, id.nvar)
  if iconf.mode == :mono
    regu = Regularization(T(sp.ρ0), T(sp.δ0), T(sp.ρ_min), T(sp.δ_min), :classic)
    D .= -T(1.0e0) / 2
  else
    regu = Regularization(T(sp.ρ0), T(sp.δ0), T(sp.ρ_min), T(sp.δ_min), :hybrid)
    D .= -T(1.0e-2)
  end
  δv = [regu.δ] # put it in a Vector so that we can modify it without modifying opK2prod!
  if sp.form_mat
    diag_Q = get_diag_Q(fd.Q)
    K = create_K2(id, D, fd.Q.data, fd.A, diag_Q, regu, fd.uplo, T)
    diagind_K = get_diagind_K(K, sp.uplo)
    if sp.equilibrate
      Deq = Diagonal(Vector{T}(undef, id.nvar + id.ncon))
      Deq.diag .= one(T)
      C_eq = Diagonal(Vector{T}(undef, id.nvar + id.ncon))
    else
      Deq = Diagonal(Vector{T}(undef, 0))
      C_eq = Diagonal(Vector{T}(undef, 0))
    end
    mt = MatrixTools(diag_Q, diagind_K, Deq, C_eq)
  else
    K = LinearOperator(
      T,
      id.nvar + id.ncon,
      id.nvar + id.ncon,
      true,
      true,
      (res, v, α, β) -> opK2prod!(res, id.nvar, fd.Q, D, fd.A, δv, v, α, β, fd.uplo),
    )
    mt = 0
  end

  rhs = similar(fd.c, id.nvar + id.ncon)
  KS = @timeit_debug to "krylov solver setup" init_Ksolver(K, rhs, sp)
  pdat = @timeit_debug to "preconditioner setup" PreconditionerData(sp, id, fd, regu, D, K)

  return PreallocatedDataK2Krylov(
    pdat,
    D,
    rhs,
    sp.rhs_scale,
    sp.equilibrate,
    regu,
    δv,
    K, #K
    mt,
    KS,
    0,
    T(sp.atol0),
    T(sp.rtol0),
    T(sp.atol_min),
    T(sp.rtol_min),
    sp.itmax,
  )
end

function solver!(
  dd::AbstractVector{T},
  pad::PreallocatedDataK2Krylov{T},
  dda::DescentDirectionAllocs{T},
  pt::Point{T},
  itd::IterData{T},
  fd::Abstract_QM_FloatData{T},
  id::QM_IntData,
  res::AbstractResiduals{T},
  cnts::Counters,
  step::Symbol,
) where {T <: Real}
  pad.rhs .= pad.equilibrate ? dd .* pad.mt.Deq.diag : dd
  if pad.rhs_scale
    rhsNorm = kscale!(pad.rhs)
  end
  if step !== :cc
    out = @timeit_debug to "preconditioner update" update_preconditioner!(
      pad.pdat,
      pad,
      itd,
      pt,
      id,
      fd,
      cnts,
    )
    pad.kiter = 0
    out == 1 && return out
  end
  @timeit_debug to "Krylov solve" ksolve!(
    pad.KS,
    pad.K,
    pad.rhs,
    pad.pdat.P,
    verbose = 0,
    atol = pad.atol,
    rtol = pad.rtol,
    itmax = pad.itmax,
  )
  pad.kiter += niterations(pad.KS)
  update_kresiduals_history!(res, pad.K, pad.KS.x, pad.rhs)
  if pad.rhs_scale
    kunscale!(pad.KS.x, rhsNorm)
  end
  if pad.equilibrate
    if typeof(pad.K) <: Symmetric{T, <:Union{SparseMatrixCSC{T}, SparseMatrixCOO{T}}} &&
       step !== :aff
      rdiv!(pad.K.data, pad.mt.Deq)
      ldiv!(pad.mt.Deq, pad.K.data)
    end
    pad.KS.x .*= pad.mt.Deq.diag
  end
  dd .= pad.KS.x

  return 0
end

function update_pad!(
  pad::PreallocatedDataK2Krylov{T},
  dda::DescentDirectionAllocs{T},
  pt::Point{T},
  itd::IterData{T},
  fd::Abstract_QM_FloatData{T},
  id::QM_IntData,
  res::AbstractResiduals{T},
  cnts::Counters,
) where {T <: Real}
  if cnts.k != 0
    update_regu!(pad.regu)
  end

  if pad.atol > pad.atol_min
    pad.atol /= 10
  end
  if pad.rtol > pad.rtol_min
    pad.rtol /= 10
  end

  pad.D .= -pad.regu.ρ
  pad.D[id.ilow] .-= pt.s_l ./ itd.x_m_lvar
  pad.D[id.iupp] .-= pt.s_u ./ itd.uvar_m_x
  pad.δv[1] = pad.regu.δ
  if typeof(pad.K) <: Symmetric{T, <:Union{SparseMatrixCSC{T}, SparseMatrixCOO{T}}}
    pad.D[pad.mt.diag_Q.nzind] .-= pad.mt.diag_Q.nzval
    update_diag_K11!(pad.K, pad.D, pad.mt.diagind_K, id.nvar)
    update_diag_K22!(pad.K, pad.regu.δ, pad.mt.diagind_K, id.nvar, id.ncon)
    if pad.equilibrate
      pad.mt.Deq.diag .= one(T)
      @timeit_debug to "equilibration" equilibrate!(
        pad.K,
        pad.mt.Deq,
        pad.mt.C_eq;
        ϵ = T(1.0e-2),
        max_iter = 15,
      )
    end
  end

  return 0
end
