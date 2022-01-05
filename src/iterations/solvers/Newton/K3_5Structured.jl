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
export K3_5StructuredParams

"""
Type to use the K3.5 formulation with a Krylov method, using the package 
[`Krylov.jl`](https://github.com/JuliaSmoothOptimizers/Krylov.jl). 
The outer constructor 

    K3_5StructuredParams(; uplo = :U,   kmethod = :trimr,
                         atol0 = 1.0e-4, rtol0 = 1.0e-4, 
                         atol_min = 1.0e-10, rtol_min = 1.0e-10,
                         ρ0 =  sqrt(eps()) * 1e3, δ0 = sqrt(eps()) * 1e4,
                         ρ_min = 1e4 * sqrt(eps()), δ_min = 1e4 * sqrt(eps()))

creates a [`RipQP.SolverParams`](@ref) that should be used to create a [`RipQP.InputConfig`](@ref).
The available methods are:
- `:tricg`
- `:trimr`

"""
mutable struct K3_5StructuredParams <: SolverParams
  uplo::Symbol
  kmethod::Symbol
  atol0::Float64
  rtol0::Float64
  atol_min::Float64
  rtol_min::Float64
  ρ0::Float64
  δ0::Float64
  ρ_min::Float64
  δ_min::Float64
end

function K3_5StructuredParams(;
  uplo::Symbol = :U,
  kmethod::Symbol = :trimr,
  atol0::T = 1.0e-4,
  rtol0::T = 1.0e-4,
  atol_min::T = 1.0e-10,
  rtol_min::T = 1.0e-10,
  ρ0::T = sqrt(eps()) * 1e3, 
  δ0::T = sqrt(eps()) * 1e4,
  ρ_min::T = 1e4 * sqrt(eps()),
  δ_min::T = 1e4 * sqrt(eps()),
) where {T <: Real}
  return K3_5StructuredParams(uplo, kmethod, atol0, rtol0, atol_min, rtol_min, ρ0, δ0, ρ_min, δ_min)
end

mutable struct PreallocatedDataK3_5Structured{
  T <: Real,
  S,
  L1 <: LinearOperator,
  L2 <: LinearOperator,
  L3 <: LinearOperator,
  Ksol <: KrylovSolver,
} <: PreallocatedDataNewtonStructured{T, S}
  rhs1::S
  rhs2::S
  regu::Regularization{T}
  δv::Vector{T}
  As::L1
  Qreg::AbstractMatrix{T} # regularized Q
  QregF::LDLFactorizations.LDLFactorization{T, Int, Int, Int}
  Qregop::L2 # factorized matrix Qreg
  opBR::L3
  KS::Ksol
  atol::T
  rtol::T
  atol_min::T
  rtol_min::T
end

function opAsprod!(
  res::AbstractVector{T},
  nvar::Int,
  ncon::Int,
  ilow::Vector{Int},
  iupp::Vector{Int},
  nlow::Int,
  s_l::AbstractVector{T},
  s_u::AbstractVector{T},
  A::AbstractMatrix{T},
  v::AbstractVector{T},
  α::T,
  β::T,
  uplo::Symbol,
) where {T}
  if β == 0
    res[(ncon + 1):(ncon + nlow)] .= @views α .* sqrt.(s_l) .* v[ilow]
    res[(ncon + nlow + 1):end] .= @views (-α) .* sqrt.(s_u) .* v[iupp]
  else
    res[(ncon + 1):(ncon + nlow)] .=
      @views α .* sqrt.(s_l) .* v[ilow] .+ β .* res[(ncon + 1):(ncon + nlow)]
    res[(ncon + nlow + 1):end] .=
      @views (-α) .* sqrt.(s_u) .* v[iupp] .+ β .* res[(ncon + nlow + 1):end]
  end
  if uplo == :U
    @views mul!(res[1:ncon], A', v, α, β)
  else
    @views mul!(res[1:ncon], A, v, α, β)
  end
end

function opAstprod!(
  res::AbstractVector{T},
  nvar::Int,
  ncon::Int,
  ilow::Vector{Int},
  iupp::Vector{Int},
  nlow::Int,
  s_l::AbstractVector{T},
  s_u::AbstractVector{T},
  A::AbstractMatrix{T},
  v::AbstractVector{T},
  α::T,
  β::T,
  uplo::Symbol,
) where {T}
  if uplo == :U
    @views mul!(res, A, v[1:ncon], α, β)
  else
    @views mul!(res, A', v[1:ncon], α, β)
  end
  res[ilow] .+= @views α .* sqrt.(s_l) .* v[(ncon + 1):(ncon + nlow)]
  res[iupp] .-= @views α .* sqrt.(s_u) .* v[(ncon + nlow + 1):end]
end

function opBRprod!(
  res::AbstractVector{T},
  ncon::Int,
  nlow::Int,
  x_m_lvar::AbstractVector{T},
  uvar_m_x::AbstractVector{T},
  δv::AbstractVector{T},
  v::AbstractVector{T},
  α::T,
  β::T,
) where {T <: Real}
  if β == zero(T)
    res[1:ncon] .= @views (α / δv[1]) .* v[1:ncon]
    res[(ncon + 1):(ncon + nlow)] .= @views α ./ x_m_lvar .* v[(ncon + 1):(ncon + nlow)]
    res[(ncon + nlow + 1):end] .= @views α ./ uvar_m_x .* v[(ncon + nlow + 1):end]
  else
    res[1:ncon] .= @views (α / δv[1]) .* v[1:ncon] .+ β .* res[1:ncon]
    res[(ncon + 1):(ncon + nlow)] .=
      @views α ./ x_m_lvar .* v[(ncon + 1):(ncon + nlow)] .+ β .* res[(ncon + 1):(ncon + nlow)]
    res[(ncon + nlow + 1):end] .=
      @views α ./ uvar_m_x .* v[(ncon + nlow + 1):end] .+ β .* res[(ncon + nlow + 1):end]
  end
end

function update_kresiduals_history!(
  res::AbstractResiduals{T},
  Qreg::AbstractMatrix{T},
  As::AbstractLinearOperator{T},
  δ::T,
  solx::AbstractVector{T},
  soly::AbstractVector{T},
  rhs1::AbstractVector{T},
  rhs2::AbstractVector{T},
  s_l::AbstractVector{T},
  s_u::AbstractVector{T},
  x_m_lvar::AbstractVector{T},
  uvar_m_x::AbstractVector{T},
  nvar::Int,
  ncon::Int,
  nlow::Int,
  ilow::Vector{Int},
  iupp::Vector{Int},
) where {T <: Real}
  if typeof(res) <: ResidualsHistory
    @views mul!(res.Kres[1:nvar], Symmetric(Qreg, :U), solx)
    @views mul!(res.Kres[1:nvar], As', soly, one(T), one(T))
    @views mul!(res.Kres[(nvar + 1):end], As, solx)
    res.Kres[(nvar + 1):(nvar + ncon)] .+= δ .* soly[1:ncon]
    res.Kres[(nvar + ncon + 1):(nvar + ncon + nlow)] .+=
      @views s_l .* solx[ilow] .+ x_m_lvar .* soly[(ncon + 1):(ncon + nlow)]
    res.Kres[(nvar + ncon + nlow + 1):end] .+=
      @views .-s_u .* solx[iupp] .+ uvar_m_x .* soly[(ncon + nlow + 1):end]
    res.Kres[1:nvar] .-= rhs1
    res.Kres[(nvar + 1):end] .-= rhs2
  end
end

function PreallocatedData(
  sp::K3_5StructuredParams,
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
    regu = Regularization(
      T(sp.ρ0),
      T(sp.δ0),
      T(sqrt(eps(T)) * 1e0),
      T(sqrt(eps(T)) * 1e0),
      :classic,
    )
  end
  δv = [regu.δ] # put it in a Vector so that we can modify it without modifying opK2prod!
  As = LinearOperator(
    T,
    id.ncon + id.nlow + id.nupp,
    id.nvar,
    false,
    false,
    (res, v, α, β) -> opAsprod!(
      res,
      id.nvar,
      id.ncon,
      id.ilow,
      id.iupp,
      id.nlow,
      pt.s_l,
      pt.s_u,
      fd.A,
      v,
      α,
      β,
      fd.uplo,
    ),
    (res, v, α, β) -> opAstprod!(
      res,
      id.nvar,
      id.ncon,
      id.ilow,
      id.iupp,
      id.nlow,
      pt.s_l,
      pt.s_u,
      fd.A,
      v,
      α,
      β,
      fd.uplo,
    ),
  )

  Qreg = fd.Q + regu.ρ_min * I
  QregF = ldl(Symmetric(Qreg, :U))
  Qregop = LinearOperator(T, id.nvar, id.nvar, true, true, (res, v) -> ldiv!(res, QregF, v))
  Rd = δv[1] * I(id.ncon)
  opBR = LinearOperator(
    T,
    id.ncon + id.nlow + id.nupp,
    id.ncon + id.nlow + id.nupp,
    true,
    true,
    (res, v, α, β) -> opBRprod!(res, id.ncon, id.nlow, itd.x_m_lvar, itd.uvar_m_x, δv, v, α, β),
  )
  rhs1 = similar(fd.c, id.nvar)
  rhs2 = similar(fd.c, id.ncon + id.nlow + id.nupp)
  kstring = string(sp.kmethod)
  KS = eval(KSolver(sp.kmethod))(As', rhs1)

  return PreallocatedDataK3_5Structured(
    rhs1,
    rhs2,
    regu,
    δv,
    As,
    Qreg,
    QregF,
    Qregop,
    opBR,
    KS,
    sp.atol0,
    sp.rtol0,
    sp.atol_min,
    sp.rtol_min,
  )
end

function solver!(
  dd::AbstractVector{T},
  pad::PreallocatedDataK3_5Structured{T},
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
  Δs_l = itd.Δs_l
  Δs_u = itd.Δs_u
  pad.rhs1 .= @views step == :init ? fd.c : dd[1:(id.nvar)]
  if step == :init && all(pad.rhs1 .== zero(T))
    pad.rhs1 .= one(T)
  end
  pad.rhs2[1:(id.ncon)] .=
    @views (step == :init && all(dd[(id.nvar + 1):end] .== zero(T))) ? one(T) :
           dd[(id.nvar + 1):end]
  pad.rhs2[(id.ncon + 1):(id.ncon + id.nlow)] .= (step == :init) ? one(T) : Δs_l ./ sqrt.(pt.s_l)
  pad.rhs2[(id.ncon + id.nlow + 1):end] .= (step == :init) ? one(T) : Δs_u ./ sqrt.(pt.s_u)
  ksolve!(
    pad.KS,
    pad.As',
    pad.rhs1,
    pad.rhs2,
    pad.Qregop,
    pad.opBR,
    verbose = 0,
    atol = pad.atol,
    rtol = pad.rtol,
  )
  update_kresiduals_history!(
    res,
    pad.Qreg,
    pad.As,
    pad.regu.δ,
    pad.KS.x,
    pad.KS.y,
    pad.rhs1,
    pad.rhs2,
    pt.s_l,
    pt.s_u,
    itd.x_m_lvar,
    itd.uvar_m_x,
    id.nvar,
    id.ncon,
    id.nlow,
    id.ilow,
    id.iupp,
  )

  dd[1:(id.nvar)] .= @views pad.KS.x
  dd[(id.nvar + 1):end] .= @views pad.KS.y[1:(id.ncon)]
  Δs_l .= @views pad.KS.y[(id.ncon + 1):(id.ncon + id.nlow)] .* sqrt.(pt.s_l)
  Δs_u .= @views pad.KS.y[(id.ncon + id.nlow + 1):end] .* sqrt.(pt.s_u)

  return 0
end

# function update_upper_Qreg!(Qreg, Q, ρ)
#   n = size(Qreg, 1)
#   nnzQ = nnz(Q)
#   if nnzQ > 0
#     for i=1:n
#       diag_idx = Qreg.colptr[i+1] - 1
#       ptrQ = Q.colptr[i+1] - 1
#       Qreg.nzval[diag_idx] = (ptrQ ≤ nnzQ && Q.rowval[ptrQ] == i) ? Q.nzval[ptrQ] + ρ : ρ
#     end
#   end
# end

function update_pad!(
  pad::PreallocatedDataK3_5Structured{T},
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
  # if cnts.k == 4
  # update_upper_Qreg!(pad.Qreg, fd.Q, pad.regu.ρ)
  # pad.Qreg[diagind(pad.Qreg)] .= fd.Q[diagind(fd.Q)] .+ pad.regu.ρ
  # ldl_factorize!(Symmetric(pad.Qreg, :U), pad.QregF)
  # end

  if pad.atol > pad.atol_min
    pad.atol /= 10
  end
  if pad.rtol > pad.rtol_min
    pad.rtol /= 10
  end

  pad.δv[1] = pad.regu.δ

  return 0
end
