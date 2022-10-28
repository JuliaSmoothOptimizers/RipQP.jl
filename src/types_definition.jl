import Base: convert

export InputTol, SolveMethod, SystemWrite, SolverParams, PreallocatedData

# problem: min 1/2 x'Qx + c'x + c0     s.t.  Ax = b,  lvar ≤ x ≤ uvar
abstract type Abstract_QM_FloatData{
  T <: Real,
  S,
  M1 <: Union{AbstractMatrix{T}, AbstractLinearOperator{T}},
  M2 <: Union{AbstractMatrix{T}, AbstractLinearOperator{T}},
} end

mutable struct QM_FloatData{T <: Real, S, M1, M2} <: Abstract_QM_FloatData{T, S, M1, M2}
  Q::M1 # size nvar * nvar
  A::M2 # size ncon * nvar, using Aᵀ is easier to form systems
  b::S # size ncon
  c::S # size nvar
  c0::T
  lvar::S # size nvar
  uvar::S # size nvar
  uplo::Symbol
end

mutable struct QM_IntData
  ilow::Vector{Int} # indices of finite elements in lvar
  iupp::Vector{Int} # indices of finite elements in uvar
  irng::Vector{Int} # indices of finite elements in both lvar and uvar
  ifree::Vector{Int} # indices of infinite elements in both lvar and uvar
  ifix::Vector{Int}
  ncon::Int # number of equality constraints after SlackModel! (= size of b)
  nvar::Int # number of variables
  nlow::Int # length(ilow)
  nupp::Int # length(iupp)
end

"""
Abstract type for tuning the parameters of the different solvers. 
Each solver has its own `SolverParams` type.
"""
abstract type SolverParams{T} end

"""
Type to write the matrix (.mtx format) and the right hand side (.rhs format) of the system to solve at each iteration.

- `write::Bool`: activate/deactivate writing of the system 
- `name::String`: name of the sytem to solve 
- `kfirst::Int`: first iteration where a system should be written
- `kgap::Int`: iteration gap between two problem writings

The constructor

    SystemWrite(; write = false, name = "", kfirst = 0, kgap = 1)

returns a `SystemWrite` structure that should be used to tell RipQP to save the system. 
See the tutorial for more information. 
"""
struct SystemWrite
  write::Bool
  name::String
  kfirst::Int
  kgap::Int
end

SystemWrite(; write::Bool = false, name::String = "", kfirst::Int = 0, kgap::Int = 1) =
  SystemWrite(write, name, kfirst, kgap)

abstract type SolveMethod end

mutable struct InputConfig{
  I <: Integer,
  SP <: SolverParams,
  SP2 <: Union{Nothing, SolverParams},
  SP3 <: Union{Nothing, SolverParams},
  SM <: SolveMethod,
}
  mode::Symbol
  early_multi_stop::Bool # stop earlier in multi-precision, based on some quantities of the algorithm
  scaling::Bool
  presolve::Bool
  normalize_rtol::Bool # normalize the primal and dual tolerance to the initial starting primal and dual residuals
  kc::I # multiple centrality corrections, -1 = automatic computation
  perturb::Bool
  minimize::Bool

  # Functions to choose formulations
  sp::SP
  sp2::SP2 # second solver to use (usually when using multi-prec in Floa64)
  sp3::SP3 # third solver to use (usually when using multi-prec in Floa128)
  solve_method::SM

  # output tools
  history::Bool
  w::SystemWrite # write systems 
end

"""
Type to specify the tolerances used by RipQP.

- `max_iter :: Int`: maximum number of iterations
- `ϵ_pdd`: relative primal-dual difference tolerance
- `ϵ_rb`: primal tolerance
- `ϵ_rc`: dual tolerance
- `max_iter1`, `ϵ_pdd1`, `ϵ_rb1`, `ϵ_rc1`: same as `max_iter`, `ϵ_pdd`, `ϵ_rb` and
    `ϵ_rc`, but used for switching from `sp1` to `sp2` (or from single to double precision if `sp2` is `nothing`).
    They are only usefull when `mode=:multi`
- `max_iter2`, `ϵ_pdd2`, `ϵ_rb2`, `ϵ_rc2`: same as `max_iter`, `ϵ_pdd`, `ϵ_rb` and
    `ϵ_rc`, but used for switching from `sp2` to `sp3` (or from double to quadruple precision if `sp3` is `nothing`).
    They are only usefull when `mode=:multi` and/or `T0=Float128`
- `ϵ_rbz` : primal transition tolerance for the zoom procedure, (used only if `refinement=:zoom`)
- `ϵ_Δx`: step tolerance for the current point estimate (note: this criterion
    is currently disabled)
- `ϵ_μ`: duality measure tolerance (note: this criterion is currently disabled)
- `max_time`: maximum time to solve the QP

The constructor

    itol = InputTol(::Type{T};
                    max_iter :: I = 200, max_iter1 :: I = 40, max_iter2 :: I = 180, 
                    ϵ_pdd :: T = 1e-8, ϵ_pdd1 :: T = 1e-2, ϵ_pdd2 :: T = 1e-4, 
                    ϵ_rb :: T = 1e-6, ϵ_rb1 :: T = 1e-4, ϵ_rb2 :: T = 1e-5, ϵ_rbz :: T = 1e-3,
                    ϵ_rc :: T = 1e-6, ϵ_rc1 :: T = 1e-4, ϵ_rc2 :: T = 1e-5,
                    ϵ_Δx :: T = 1e-16, ϵ_μ :: T = 1e-9) where {T<:Real, I<:Integer}

    InputTol(; kwargs...) = InputTol(Float64; kwargs...) 

returns a `InputTol` struct that initializes the stopping criteria for RipQP. 
The 1 and 2 characters refer to the transitions between the chosen solvers in `:multi`.
If `sp2` and `sp3` are not precised when calling [`RipQP.ripqp`](@ref),
they refer to transitions between floating-point systems.
"""
struct InputTol{T <: Real, I <: Integer}
  # maximum number of iterations
  max_iter::I
  max_iter1::I # only in multi mode
  max_iter2::I # only in multi mode with T0 = Float128

  # relative primal-dual gap tolerance
  ϵ_pdd::T
  ϵ_pdd1::T # only in multi mode 
  ϵ_pdd2::T # only in multi mode with T0 = Float128

  # primal residual
  ϵ_rb::T
  ϵ_rb1::T # only in multi mode 
  ϵ_rb2::T # only in multi mode with T0 = Float128
  ϵ_rbz::T # only when using zoom refinement

  # dual residual
  ϵ_rc::T
  ϵ_rc1::T # only in multi mode 
  ϵ_rc2::T # only in multi mode with T0 = Float128

  # unused residuals (for now)
  ϵ_μ::T
  ϵ_Δx::T

  # maximum time for resolution
  max_time::Float64
end

function InputTol(
  ::Type{T};
  max_iter::I = 200,
  max_iter1::I = 40,
  max_iter2::I = 180,
  ϵ_pdd::T = (T == Float64) ? 1e-8 : sqrt(eps(T)),
  ϵ_pdd1::T = T(1e-2),
  ϵ_pdd2::T = T(1e-4),
  ϵ_rb::T = (T == Float64) ? 1e-6 : sqrt(eps(T)),
  ϵ_rb1::T = T(1e-4),
  ϵ_rb2::T = T(1e-5),
  ϵ_rbz::T = T(1e-5),
  ϵ_rc::T = (T == Float64) ? 1e-6 : sqrt(eps(T)),
  ϵ_rc1::T = T(1e-4),
  ϵ_rc2::T = T(1e-5),
  ϵ_Δx::T = eps(T),
  ϵ_μ::T = sqrt(eps(T)),
  max_time::Float64 = 1200.0,
) where {T <: Real, I <: Integer}
  return InputTol{T, I}(
    max_iter,
    max_iter1,
    max_iter2,
    ϵ_pdd,
    ϵ_pdd1,
    ϵ_pdd2,
    ϵ_rb,
    ϵ_rb1,
    ϵ_rb2,
    ϵ_rbz,
    ϵ_rc,
    ϵ_rc1,
    ϵ_rc2,
    ϵ_μ,
    ϵ_Δx,
    max_time,
  )
end

InputTol(; kwargs...) = InputTol(Float64; kwargs...)

mutable struct Tolerances{T <: Real}
  pdd::T  # primal-dual difference (relative)
  rb::T  # primal residuals tolerance
  rc::T  # dual residuals tolerance
  tol_rb::T  # ϵ_rb * (1 + ||r_b0||)
  tol_rc::T  # ϵ_rc * (1 + ||r_c0||)
  μ::T  # duality measure
  Δx::T
  normalize_rtol::Bool # true if normalize_rtol=true, then tol_rb, tol_rc = ϵ_rb, ϵ_rc
end

mutable struct Point{T <: Real, S}
  x::S # size nvar
  y::S # size ncon
  s_l::S # size nlow (useless zeros corresponding to infinite lower bounds are not stored)
  s_u::S # size nupp (useless zeros corresponding to infinite upper bounds are not stored)
  function Point(
    x::AbstractVector{T},
    y::AbstractVector{T},
    s_l::AbstractVector{T},
    s_u::AbstractVector{T},
  ) where {T <: Real}
    S = typeof(x)
    return new{T, S}(x, y, s_l, s_u)
  end
end

convert(::Type{Point{T, S}}, pt::Point) where {T <: Real, S} =
  Point(convert(S, pt.x), convert(S, pt.y), convert(S, pt.s_l), convert(S, pt.s_u))

abstract type AbstractResiduals{T <: Real, S} end

mutable struct Residuals{T <: Real, S} <: AbstractResiduals{T, S}
  rb::S # primal residuals Ax - b
  rc::S # dual residuals -Qx + Aᵀy + s_l - s_u
  rbNorm::T # ||rb||
  rcNorm::T # ||rc||
end

convert(::Type{AbstractResiduals{T, S}}, res::Residuals) where {T <: Real, S <: AbstractVector{T}} =
  Residuals(convert(S, res.rb), convert(S, res.rc), convert(T, res.rbNorm), convert(T, res.rcNorm))

mutable struct ResidualsHistory{T <: Real, S} <: AbstractResiduals{T, S}
  rb::S # primal residuals Ax - b
  rc::S # dual residuals -Qx + Aᵀy + s_l - s_u
  rbNorm::T # ||rb||
  rcNorm::T # ||rc||
  rbNormH::Vector{T} # list of rb values
  rcNormH::Vector{T} # list of rc values
  pddH::Vector{T} # list of pdd values
  kiterH::Vector{Int} # number of matrix vector product if using a Krylov method
  μH::Vector{T} # list of μ values
  min_bound_distH::Vector{T} # list of minimum values of x - lvar and uvar - x
  KΔxy::S # K * Δxy
  Kres::S # ||KΔxy-rhs|| (residuals Krylov method)
  KresNormH::Vector{T} # list of ||KΔxy-rhs||
  KresPNormH::Vector{T}
  KresDNormH::Vector{T}
end

convert(
  ::Type{AbstractResiduals{T, S}},
  res::ResidualsHistory,
) where {T <: Real, S <: AbstractVector{T}} = ResidualsHistory(
  convert(S, res.rb),
  convert(S, res.rc),
  convert(T, res.rbNorm),
  convert(T, res.rcNorm),
  convert(Array{T, 1}, res.rbNormH),
  convert(Array{T, 1}, res.rcNormH),
  convert(Array{T, 1}, res.pddH),
  res.kiterH,
  convert(Array{T, 1}, res.μH),
  convert(Array{T, 1}, res.min_bound_distH),
  convert(S, res.KΔxy),
  convert(S, res.Kres),
  convert(Array{T, 1}, res.KresNormH),
  convert(Array{T, 1}, res.KresPNormH),
  convert(Array{T, 1}, res.KresDNormH),
)

function init_residuals(
  rb::AbstractVector{T},
  rc::AbstractVector{T},
  rbNorm::T,
  rcNorm::T,
  iconf::InputConfig,
  id::QM_IntData,
) where {T <: Real}
  S = typeof(rb)
  if iconf.history
    stype = typeof(iconf.sp)
    if stype <: NewtonParams
      Kn = length(rb) + length(rc) + id.nlow + id.nupp
    elseif stype <: NormalParams
      Kn = length(rb)
    else
      Kn = length(rb) + length(rc)
    end
    KΔxy = S(undef, Kn)
    Kres = S(undef, Kn)
    return ResidualsHistory{T, S}(
      rb,
      rc,
      rbNorm,
      rcNorm,
      T[],
      T[],
      T[],
      Int[],
      T[],
      T[],
      KΔxy,
      Kres,
      T[],
      T[],
      T[],
    )
  else
    return Residuals{T, S}(rb, rc, rbNorm, rcNorm)
  end
end

mutable struct Regularization{T <: Real}
  ρ::T       # curent top-left regularization parameter
  δ::T       # cureent bottom-right regularization parameter
  ρ_min::T       # ρ minimum value 
  δ_min::T       # δ minimum value 
  regul::Symbol  # regularization mode (:classic, :dynamic, or :none)
end

convert(::Type{Regularization{T}}, regu::Regularization{T0}) where {T <: Real, T0 <: Real} =
  Regularization(T(regu.ρ), T(regu.δ), T(regu.ρ_min), T(regu.δ_min), regu.regul)

abstract type IterData{T <: Real, S} end

mutable struct IterDataCPU{T <: Real, S} <: IterData{T, S}
  Δxy::S # Newton step [Δx; Δy]
  Δs_l::S
  Δs_u::S
  x_m_lvar::S # x - lvar
  uvar_m_x::S # uvar - x
  Qx::S
  ATy::S # Aᵀy
  Ax::S
  xTQx_2::T # xᵀQx
  cTx::T # cᵀx
  pri_obj::T # 1/2 xᵀQx + cᵀx + c0                                             
  dual_obj::T # -1/2 xᵀQx + yᵀb + s_lᵀlvar - s_uᵀuvar + c0
  μ::T # duality measure (s_lᵀ(x-lvar) + s_uᵀ(uvar-x)) / (nlow+nupp)
  pdd::T # primal dual difference (relative) pri_obj - dual_obj / pri_obj
  qp::Bool # true if qp false if lp
  minimize::Bool
  perturb::Bool
end

mutable struct IterDataGPU{T <: Real, S} <: IterData{T, S}
  Δxy::S # Newton step [Δx; Δy]
  Δs_l::S
  Δs_u::S
  x_m_lvar::S # x - lvar
  uvar_m_x::S # uvar - x
  Qx::S
  ATy::S # Aᵀy
  Ax::S
  xTQx_2::T # xᵀQx
  cTx::T # cᵀx
  pri_obj::T # 1/2 xᵀQx + cᵀx + c0                                             
  dual_obj::T # -1/2 xᵀQx + yᵀb + s_lᵀlvar - s_uᵀuvar + c0
  μ::T # duality measure (s_lᵀ(x-lvar) + s_uᵀ(uvar-x)) / (nlow+nupp)
  pdd::T # primal dual difference (relative) pri_obj - dual_obj / pri_obj
  qp::Bool # true if qp false if lp
  minimize::Bool
  perturb::Bool
  store_vpri::S
  store_vdual_l::S
  store_vdual_u::S
  IterDataGPU(
    Δxy::S,
    Δs_l::S,
    Δs_u::S,
    x_m_lvar::S,
    uvar_m_x::S,
    Qx::S,
    ATy::S,
    Ax::S,
    xTQx_2::T,
    cTx::T,
    pri_obj::T,
    dual_obj::T,
    μ::T,
    pdd::T,
    qp::Bool,
    minimize::Bool,
    perturb::Bool,
  ) where {T <: Real, S} = new{T, S}(
    Δxy,
    Δs_l,
    Δs_u,
    x_m_lvar,
    uvar_m_x,
    Qx,
    ATy,
    Ax,
    xTQx_2,
    cTx,
    pri_obj,
    dual_obj,
    μ,
    pdd,
    qp,
    minimize,
    perturb,
    similar(Qx),
    similar(Δs_l),
    similar(Δs_u),
  )
end

function IterData(
  Δxy,
  Δs_l,
  Δs_u,
  x_m_lvar,
  uvar_m_x,
  Qx,
  ATy,
  Ax,
  xTQx_2,
  cTx,
  pri_obj,
  dual_obj,
  μ,
  pdd,
  qp,
  minimize,
  perturb,
)
  if typeof(Δxy) <: Vector
    return IterDataCPU(
      Δxy,
      Δs_l,
      Δs_u,
      x_m_lvar,
      uvar_m_x,
      Qx,
      ATy,
      Ax,
      xTQx_2,
      cTx,
      pri_obj,
      dual_obj,
      μ,
      pdd,
      qp,
      minimize,
      perturb,
    )
  else
    return IterDataGPU(
      Δxy,
      Δs_l,
      Δs_u,
      x_m_lvar,
      uvar_m_x,
      Qx,
      ATy,
      Ax,
      xTQx_2,
      cTx,
      pri_obj,
      dual_obj,
      μ,
      pdd,
      qp,
      minimize,
      perturb,
    )
  end
end

convert(
  ::Type{IterData{T, S}},
  itd::IterData{T0, S0},
) where {T <: Real, S <: AbstractVector{T}, T0 <: Real, S0} = IterData(
  convert(S, itd.Δxy),
  convert(S, itd.Δs_l),
  convert(S, itd.Δs_u),
  convert(S, itd.x_m_lvar),
  convert(S, itd.uvar_m_x),
  convert(S, itd.Qx),
  convert(S, itd.ATy),
  convert(S, itd.Ax),
  convert(T, itd.xTQx_2),
  convert(T, itd.cTx),
  convert(T, itd.pri_obj),
  convert(T, itd.dual_obj),
  convert(T, itd.μ),
  convert(T, itd.pdd),
  itd.qp,
  itd.minimize,
  itd.perturb,
)

abstract type ScaleData{T <: Real, S} end

mutable struct ScaleDataLP{T <: Real, S} <: ScaleData{T, S}
  d1::S
  d2::S
  r_k::S
  c_k::S
end

mutable struct ScaleDataQP{T <: Real, S} <: ScaleData{T, S}
  deq::S
  c_k::S
end

mutable struct StartingPointData{T <: Real, S}
  dual_val::S
  s0_l1::S
  s0_u1::S
end

function ScaleData(fd::QM_FloatData{T, S}, id::QM_IntData, scaling::Bool) where {T, S}
  if scaling
    if nnz(fd.Q) > 0
      sd =
        ScaleDataQP{T, S}(fill!(S(undef, id.nvar + id.ncon), one(T)), S(undef, id.nvar + id.ncon))
    else
      if fd.uplo == :U
        m, n = id.nvar, id.ncon
      else
        m, n = id.ncon, id.nvar
      end
      sd = ScaleDataLP{T, S}(
        fill!(S(undef, id.nvar), one(T)),
        fill!(S(undef, id.ncon), one(T)),
        S(undef, n),
        S(undef, m),
      )
    end
  else
    empty_v = S(undef, 0)
    sd = ScaleDataQP{T, S}(empty_v, empty_v)
  end
end

convert(
  ::Type{StartingPointData{T, S}},
  spd::StartingPointData{T0, S0},
) where {T, S <: AbstractVector{T}, T0, S0} =
  StartingPointData{T, S}(convert(S, spd.dual_val), convert(S, spd.s0_l1), convert(S, spd.s0_u1))

abstract type PreallocatedData{T <: Real, S} end

mutable struct StopCrit{T}
  optimal::Bool
  small_μ::Bool
  tired::Bool
  max_iter::Int
  max_time::T
  start_time::T
  Δt::T
end

mutable struct Counters
  c_catch::Int # safety try:cath
  c_regu_dim::Int # number of δ_min reductions
  k::Int # iter count
  km::Int # iter relative to precision: if k+=1 and T==Float128, km +=16  (km+=4 if T==Float64 and km+=1 if T==Float32)
  tfact::UInt64
  tsolve::UInt64
  kc::Int # maximum corrector steps
  c_ref::Int # current number of refinements
  w::SystemWrite # store SystemWrite data
  last_sp::Bool # true if currently using the last solver to iterate (always true in mono-precision)
end
