# K3.5 = D K3 D⁻¹ , 
#      [ I   0   0       0  ]
# D² = [ 0   I   0       0  ] 
#      [ 0   0  S_l⁻¹    0  ]
#      [ 0   0   0     S_u⁻¹]
#
# [-Q - ρI    Aᵀ   √S_l  -√S_u ][ Δx  ]     [    -rc        ]
# [   A       δI    0      0   ][ Δy  ]     [    -rb        ]
# [  √S_l     0    X-L     0   ][Δs_l2] = D [σμe - (X-L)S_le]
# [ -√S_u     0     0     U-X  ][Δs_u2]     [σμe + (U-X)S_ue]
export K3_5KrylovParams

"""
Type to use the K3.5 formulation with a Krylov method, using the package 
[`Krylov.jl`](https://github.com/JuliaSmoothOptimizers/Krylov.jl). 
The outer constructor 

    K3_5KrylovParams(; uplo = :L, kmethod = :minres, preconditioner = :Identity,
                     atol0 = 1.0e-4, rtol0 = 1.0e-4,
                     atol_min = 1.0e-10, rtol_min = 1.0e-10,
                     ρ0 = sqrt(eps()) * 1e5, δ0 = sqrt(eps()) * 1e5,
                     ρ_min = 1e3 * sqrt(eps()), δ_min = 1e4 * sqrt(eps()))

creates a [`RipQP.SolverParams`](@ref) that should be used to create a [`RipQP.InputConfig`](@ref).
The available methods are:
- `:minres`
- `:minres_qlp`

"""
mutable struct K3_5KrylovParams <: SolverParams
  uplo::Symbol
  kmethod::Symbol
  preconditioner::Symbol
  atol0::Float64
  rtol0::Float64
  atol_min::Float64
  rtol_min::Float64
  ρ0::Float64
  δ0::Float64
  ρ_min::Float64
  δ_min::Float64
end

function K3_5KrylovParams(;
  uplo::Symbol = :L,
  kmethod::Symbol = :minres,
  preconditioner::Symbol = :Identity,
  atol0::T = 1.0e-4,
  rtol0::T = 1.0e-4,
  atol_min::T = 1.0e-10,
  rtol_min::T = 1.0e-10,
  ρ0::T = sqrt(eps()) * 1e5,
  δ0::T = sqrt(eps()) * 1e5,
  ρ_min::T = 1e3 * sqrt(eps()),
  δ_min::T = 1e4 * sqrt(eps()),
) where {T <: Real}
  return K3_5KrylovParams(
    uplo,
    kmethod,
    preconditioner,
    atol0,
    rtol0,
    atol_min,
    rtol_min,
    ρ0,
    δ0,
    ρ_min,
    δ_min,
  )
end

mutable struct PreallocatedDataK3_5Krylov{
  T <: Real,
  S,
  L <: LinearOperator,
  Ksol <: KrylovSolver,
} <: PreallocatedDataNewtonKrylov{T, S}
  rhs::S
  regu::Regularization{T}
  ρv::Vector{T}
  δv::Vector{T}
  K::L # augmented matrix (LinearOperator)         
  KS::Ksol
  atol::T
  rtol::T
  atol_min::T
  rtol_min::T
end

function opK3_5prod!(
  res::AbstractVector{T},
  nvar::Int,
  ncon::Int,
  ilow::Vector{Int},
  iupp::Vector{Int},
  nlow::Int,
  x_m_lvar::AbstractVector{T},
  uvar_m_x::AbstractVector{T},
  s_l::AbstractVector{T},
  s_u::AbstractVector{T},
  Q::Union{AbstractMatrix{T}, AbstractLinearOperator{T}},
  A::Union{AbstractMatrix{T}, AbstractLinearOperator{T}},
  ρv::AbstractVector{T},
  δv::AbstractVector{T},
  v::AbstractVector{T},
  α::T,
  β::T,
  uplo::Symbol,
) where {T}
  @views mul!(res[1:nvar], Q, v[1:nvar], -α, β)
  res[1:nvar] .-= @views (α * ρv[1]) .* v[1:nvar]
  res[ilow] .+= @views α .* sqrt.(s_l) .* v[(nvar + ncon + 1):(nvar + ncon + nlow)]
  res[iupp] .-= @views α .* sqrt.(s_u) .* v[(nvar + ncon + nlow + 1):end]
  if uplo == :U
    @views mul!(res[1:nvar], A, v[(nvar + 1):(nvar + ncon)], α, one(T))
    @views mul!(res[(nvar + 1):(nvar + ncon)], A', v[1:nvar], α, β)
  else
    @views mul!(res[1:nvar], A', v[(nvar + 1):(nvar + ncon)], α, one(T))
    @views mul!(res[(nvar + 1):(nvar + ncon)], A, v[1:nvar], α, β)
  end
  res[(nvar + 1):(nvar + ncon)] .+= @views (α * δv[1]) .* v[(nvar + 1):(nvar + ncon)]
  if β == 0
    res[(nvar + ncon + 1):(nvar + ncon + nlow)] .=
      @views α .* (sqrt.(s_l) .* v[ilow] .+ x_m_lvar .* v[(nvar + ncon + 1):(nvar + ncon + nlow)])
    res[(nvar + ncon + nlow + 1):end] .=
      @views α .* (.-sqrt.(s_u) .* v[iupp] .+ uvar_m_x .* v[(nvar + ncon + nlow + 1):end])
  else
    res[(nvar + ncon + 1):(nvar + ncon + nlow)] .= @views α .*
           (sqrt.(s_l) .* v[ilow] .+ x_m_lvar .* v[(nvar + ncon + 1):(nvar + ncon + nlow)]) .+
           β .* res[(nvar + ncon + 1):(nvar + ncon + nlow)]
    res[(nvar + ncon + nlow + 1):end] .=
      @views α .* (.-sqrt.(s_u) .* v[iupp] .+ uvar_m_x .* v[(nvar + ncon + nlow + 1):end]) .+
             β .* res[(nvar + ncon + nlow + 1):end]
  end
end

function PreallocatedData(
  sp::K3_5KrylovParams,
  fd::QM_FloatData{T},
  id::QM_IntData,
  itd::IterData{T},
  pt::Point{T},
  iconf::InputConfig{Tconf},
) where {T <: Real, Tconf <: Real}

  # init Regularization values
  if iconf.mode == :mono
    regu = Regularization(T(sp.ρ0), T(sp.δ0), T(sp.ρ_min), T(sp.δ_min), :classic)
  else
    regu =
      Regularization(T(sp.ρ0), T(sp.δ0), T(sqrt(eps(T)) * 1e0), T(sqrt(eps(T)) * 1e0), :classic)
  end
  ρv = [regu.ρ]
  δv = [regu.δ] # put it in a Vector so that we can modify it without modifying opK2prod!
  K = LinearOperator(
    T,
    id.nvar + id.ncon + id.nlow + id.nupp,
    id.nvar + id.ncon + id.nlow + id.nupp,
    true,
    true,
    (res, v, α, β) -> opK3_5prod!(
      res,
      id.nvar,
      id.ncon,
      id.ilow,
      id.iupp,
      id.nlow,
      itd.x_m_lvar,
      itd.uvar_m_x,
      pt.s_l,
      pt.s_u,
      fd.Q,
      fd.A,
      ρv,
      δv,
      v,
      α,
      β,
      fd.uplo,
    ),
  )

  rhs = similar(fd.c, id.nvar + id.ncon + id.nlow + id.nupp)
  kstring = string(sp.kmethod)
  KS = eval(KSolver(sp.kmethod))(K, rhs)

  return PreallocatedDataK3_5Krylov(
    rhs,
    regu,
    ρv,
    δv,
    K, #K
    KS,
    sp.atol0,
    sp.rtol0,
    sp.atol_min,
    sp.rtol_min,
  )
end

function solver!(
  dd::AbstractVector{T},
  pad::PreallocatedDataK3_5Krylov{T},
  dda::DescentDirectionAllocs{T},
  pt::Point{T},
  itd::IterData{T},
  fd::Abstract_QM_FloatData{T},
  id::QM_IntData,
  res::AbstractResiduals{T},
  cnts::Counters,
  T0::DataType,
  step::Symbol,
) where {T <: Real}
  if step == :aff
    Δs_l = dda.Δs_l_aff
    Δs_u = dda.Δs_u_aff
  else
    Δs_l = itd.Δs_l
    Δs_u = itd.Δs_u
  end
  pad.rhs[1:(id.nvar + id.ncon)] .= dd
  pad.rhs[(id.nvar + id.ncon + 1):(id.nvar + id.ncon + id.nlow)] .= Δs_l ./ sqrt.(pt.s_l)
  pad.rhs[(id.nvar + id.ncon + id.nlow + 1):end] .= Δs_u ./ sqrt.(pt.s_u)
  rhsNorm = kscale!(pad.rhs)
  pad.K.nprod = 0
  ksolve!(
    pad.KS,
    pad.K,
    pad.rhs,
    I(id.nvar + id.ncon + id.nlow + id.nupp),
    verbose = 0,
    atol = pad.atol,
    rtol = pad.rtol,
  )
  update_kresiduals_history!(res, pad.K, pad.KS.x, pad.rhs)
  kunscale!(pad.KS.x, rhsNorm)

  dd .= @views pad.KS.x[1:(id.nvar + id.ncon)]
  Δs_l .= @views pad.KS.x[(id.nvar + id.ncon + 1):(id.nvar + id.ncon + id.nlow)] .* sqrt.(pt.s_l)
  Δs_u .= @views pad.KS.x[(id.nvar + id.ncon + id.nlow + 1):end] .* sqrt.(pt.s_u)

  return 0
end

function update_pad!(
  pad::PreallocatedDataK3_5Krylov{T},
  dda::DescentDirectionAllocs{T},
  pt::Point{T},
  itd::IterData{T},
  fd::Abstract_QM_FloatData{T},
  id::QM_IntData,
  res::AbstractResiduals{T},
  cnts::Counters,
  T0::DataType,
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

  pad.ρv[1] = pad.regu.ρ
  pad.δv[1] = pad.regu.δ

  # update_preconditioner!(pad.pdat, pad, itd, pt, id, fd, cnts)

  return 0
end
