function init_Ksolver(M, v, sp::SolverParams)
  kmethod = sp.kmethod
  if kmethod ∈ (:gpmr, :diom, :fom, :dqgmres, :gmres)
    return krylov_workspace(Val(kmethod), M, v, sp.mem)
  elseif kmethod == :gmresir
    return GmresIRWorkspace(
      GmresWorkspace(M, v, sp.mem),
      similar(v, sp.Tir),
      similar(v),
      similar(v),
      false,
      0,
    )
  elseif kmethod == :ir
    return IRWorkspace(similar(v), similar(v), similar(v, sp.Tir), similar(v), similar(v), false, 0)
  else
    return krylov_workspace(Val(kmethod), M, v)
  end
end

ksolve!(
  KS::MinresWorkspace{T},
  K,
  rhs::AbstractVector{T},
  M;
  verbose::Integer = 0,
  atol::T = T(sqrt(eps(T))),
  rtol::T = T(sqrt(eps(T))),
  itmax::Int = 0,
) where {T} = minres!(KS, K, rhs, M = M, verbose = verbose, atol = atol, rtol = rtol, itmax = itmax)

ksolve!(
  KS::MinresQlpWorkspace{T},
  K,
  rhs::AbstractVector{T},
  M;
  verbose::Integer = 0,
  atol::T = T(sqrt(eps(T))),
  rtol::T = T(sqrt(eps(T))),
  itmax::Int = 0,
) where {T} =
  minres_qlp!(KS, K, rhs, M = M, verbose = verbose, atol = atol, rtol = rtol, itmax = itmax)

ksolve!(
  KS::SymmlqWorkspace{T},
  K,
  rhs::AbstractVector{T},
  M;
  verbose::Integer = 0,
  atol::T = T(sqrt(eps(T))),
  rtol::T = T(sqrt(eps(T))),
  itmax::Int = 0,
) where {T} = symmlq!(KS, K, rhs, M = M, verbose = verbose, atol = atol, rtol = rtol, itmax = itmax)

ksolve!(
  KS::CgWorkspace{T},
  K,
  rhs::AbstractVector{T},
  M;
  verbose::Integer = 0,
  atol::T = T(sqrt(eps(T))),
  rtol::T = T(sqrt(eps(T))),
  itmax::Int = 0,
) where {T} = cg!(KS, K, rhs, M = M, verbose = verbose, atol = atol, rtol = rtol, itmax = itmax)

ksolve!(
  KS::CgLanczosWorkspace{T},
  K,
  rhs::AbstractVector{T},
  M;
  verbose::Integer = 0,
  atol::T = T(sqrt(eps(T))),
  rtol::T = T(sqrt(eps(T))),
  itmax::Int = 0,
) where {T} =
  cg_lanczos!(KS, K, rhs, M = M, verbose = verbose, atol = atol, rtol = rtol, itmax = itmax)

ksolve!(
  KS::CrWorkspace{T},
  K,
  rhs::AbstractVector{T},
  M;
  verbose::Integer = 0,
  atol::T = T(sqrt(eps(T))),
  rtol::T = T(sqrt(eps(T))),
  itmax::Int = 0,
) where {T} = cr!(KS, K, rhs, M = M, verbose = verbose, atol = atol, rtol = rtol, itmax = itmax)

ksolve!(
  KS::BilqWorkspace{T},
  K,
  rhs::AbstractVector{T},
  M;
  verbose::Integer = 0,
  atol::T = T(sqrt(eps(T))),
  rtol::T = T(sqrt(eps(T))),
  itmax::Int = 0,
) where {T} = bilq!(KS, K, rhs, verbose = verbose, atol = atol, rtol = rtol, itmax = itmax)

ksolve!(
  KS::QmrWorkspace{T},
  K,
  rhs::AbstractVector{T},
  M;
  verbose::Integer = 0,
  atol::T = T(sqrt(eps(T))),
  rtol::T = T(sqrt(eps(T))),
  itmax::Int = 0,
) where {T} = qmr!(KS, K, rhs, verbose = verbose, atol = atol, rtol = rtol, itmax = itmax)

ksolve!(
  KS::UsymlqWorkspace{T},
  K,
  rhs::AbstractVector{T},
  M;
  verbose::Integer = 0,
  atol::T = T(sqrt(eps(T))),
  rtol::T = T(sqrt(eps(T))),
  itmax::Int = 0,
) where {T} = usymlq!(KS, K, rhs, rhs, verbose = verbose, atol = atol, rtol = rtol, itmax = itmax)

ksolve!(
  KS::UsymqrWorkspace{T},
  K,
  rhs::AbstractVector{T},
  M;
  verbose::Integer = 0,
  atol::T = T(sqrt(eps(T))),
  rtol::T = T(sqrt(eps(T))),
  itmax::Int = 0,
) where {T} = usymqr!(KS, K, rhs, rhs, verbose = verbose, atol = atol, rtol = rtol, itmax = itmax)

ksolve!(
  KS::BicgstabWorkspace{T},
  K,
  rhs::AbstractVector{T},
  M;
  verbose::Integer = 0,
  atol::T = T(sqrt(eps(T))),
  rtol::T = T(sqrt(eps(T))),
  itmax::Int = 0,
) where {T} = bicgstab!(KS, K, rhs, verbose = verbose, atol = atol, rtol = rtol, itmax = itmax)

ksolve!(
  KS::DiomWorkspace{T},
  K,
  rhs::AbstractVector{T},
  M;
  verbose::Integer = 0,
  atol::T = T(sqrt(eps(T))),
  rtol::T = T(sqrt(eps(T))),
  itmax::Int = 0,
) where {T} = diom!(KS, K, rhs, verbose = verbose, atol = atol, rtol = rtol, itmax = itmax)

ksolve!(
  KS::FomWorkspace{T},
  K,
  rhs::AbstractVector{T},
  M;
  verbose::Integer = 0,
  atol::T = T(sqrt(eps(T))),
  rtol::T = T(sqrt(eps(T))),
  itmax::Int = 0,
) where {T} = fom!(KS, K, rhs, verbose = verbose, atol = atol, rtol = rtol, itmax = itmax)

ksolve!(
  KS::DqgmresWorkspace{T},
  K,
  rhs::AbstractVector{T},
  M;
  verbose::Integer = 0,
  atol::T = T(sqrt(eps(T))),
  rtol::T = T(sqrt(eps(T))),
  itmax::Int = 0,
) where {T} =
  dqgmres!(KS, K, rhs, M = M, N = I, verbose = verbose, atol = atol, rtol = rtol, itmax = itmax)

ksolve!(
  KS::DqgmresWorkspace{T},
  K,
  rhs::AbstractVector{T},
  P::LRPrecond;
  verbose::Integer = 0,
  atol::T = T(sqrt(eps(T))),
  rtol::T = T(sqrt(eps(T))),
  itmax::Int = 0,
) where {T} =
  dqgmres!(KS, K, rhs, M = P.M, N = P.N, verbose = verbose, atol = atol, rtol = rtol, itmax = itmax)

ksolve!(
  KS::GmresWorkspace{T},
  K,
  rhs::AbstractVector{T},
  M;
  verbose::Integer = 0,
  atol::T = T(sqrt(eps(T))),
  rtol::T = T(sqrt(eps(T))),
  itmax::Int = 0,
) where {T} = gmres!(
  KS,
  K,
  rhs,
  M = I,
  N = M,
  restart = true,
  verbose = verbose,
  atol = atol,
  rtol = rtol,
  itmax = itmax,
)

ksolve!(
  KS::GmresWorkspace{T},
  K,
  rhs::AbstractVector{T},
  P::LRPrecond;
  verbose::Integer = 0,
  atol::T = T(sqrt(eps(T))),
  rtol::T = T(sqrt(eps(T))),
  itmax::Int = 0,
) where {T} = gmres!(
  KS,
  K,
  rhs,
  M = P.M,
  N = P.N,
  restart = true,
  verbose = verbose,
  atol = atol,
  rtol = rtol,
  itmax = itmax,
)

ksolve!(
  KS::TricgWorkspace{T},
  A,
  ξ1::AbstractVector{T},
  ξ2::AbstractVector{T},
  M,
  N;
  verbose::Integer = 0,
  atol::T = T(sqrt(eps(T))),
  rtol::T = T(sqrt(eps(T))),
  gsp::Bool = false,
  itmax::Int = 0,
) where {T} = tricg!(
  KS,
  A,
  ξ1,
  ξ2,
  M = M,
  N = N,
  flip = true,
  verbose = verbose,
  atol = atol,
  rtol = rtol,
  itmax = itmax,
)

ksolve!(
  KS::TrimrWorkspace{T},
  A,
  ξ1::AbstractVector{T},
  ξ2::AbstractVector{T},
  M,
  N;
  verbose::Integer = 0,
  atol::T = T(sqrt(eps(T))),
  rtol::T = T(sqrt(eps(T))),
  gsp::Bool = false,
  itmax::Int = 0,
) where {T} = trimr!(
  KS,
  A,
  ξ1,
  ξ2,
  M = M,
  N = (gsp == true ? I : N),
  τ = -one(T),
  ν = (gsp ? zero(T) : one(T)),
  verbose = verbose,
  atol = atol,
  rtol = rtol,
  itmax = itmax,
)

function ksolve!(
  KS::GpmrWorkspace{T},
  A,
  ξ1::AbstractVector{T},
  ξ2::AbstractVector{T},
  M,
  N;
  verbose::Integer = 0,
  atol::T = T(sqrt(eps(T))),
  rtol::T = T(sqrt(eps(T))),
  gsp::Bool = false,
  itmax::Int = 0,
) where {T}
  sqrtδI = sqrt(N.λ) * I
  return gpmr!(
    KS,
    A,
    A',
    ξ1,
    ξ2,
    C = sqrt.(M),
    D = gsp ? I : sqrtδI,
    E = sqrt.(M),
    F = gsp ? I : sqrtδI,
    λ = -one(T),
    μ = gsp ? zero(T) : one(T),
    verbose = verbose,
    atol = atol,
    rtol = rtol,
    itmax = itmax,
  )
end

# gpmr solver for K3.5
function ksolve!(
  KS::GpmrWorkspace{T},
  A,
  ξ1::AbstractVector{T},
  ξ2::AbstractVector{T},
  M,
  N::AbstractLinearOperator{T};
  verbose::Integer = 0,
  atol::T = T(sqrt(eps(T))),
  rtol::T = T(sqrt(eps(T))),
  itmax::Int = 0,
) where {T}
  return gpmr!(
    KS,
    A,
    A',
    ξ1,
    ξ2,
    C = M,
    D = N,
    E = transpose(M),
    F = N,
    λ = -one(T),
    verbose = verbose,
    atol = atol,
    rtol = rtol,
    itmax = itmax,
  )
end

function ksolve!(
  KS::LslqWorkspace{T},
  A,
  ξ1::AbstractVector{T},
  M,
  δ::T;
  verbose::Integer = 0,
  atol::T = T(sqrt(eps(T))),
  rtol::T = T(sqrt(eps(T))),
  itmax::Int = 0,
) where {T}
  return lslq!(
    KS,
    A,
    ξ1,
    M = M,
    N = δ > zero(T) ? (one(T) / δ) * I : I,
    verbose = verbose,
    atol = atol,
    btol = rtol,
    etol = zero(T),
    utol = zero(T),
    conlim = T(Inf),
    sqd = δ > zero(T),
    itmax = itmax,
  )
end

function ksolve!(
  KS::LsqrWorkspace{T},
  A,
  ξ1::AbstractVector{T},
  M,
  δ::T;
  verbose::Integer = 0,
  atol::T = T(sqrt(eps(T))),
  rtol::T = T(sqrt(eps(T))),
  itmax::Int = 0,
) where {T}
  return lsqr!(
    KS,
    A,
    ξ1,
    M = M,
    N = δ > zero(T) ? (one(T) / δ) * I : I,
    verbose = verbose,
    axtol = atol,
    btol = rtol,
    # atol = atol,
    # rtol = rtol,
    etol = zero(T),
    conlim = T(Inf),
    sqd = δ > zero(T),
    itmax = itmax,
  )
end

function ksolve!(
  KS::LsmrWorkspace{T},
  A,
  ξ1::AbstractVector{T},
  M,
  δ::T;
  verbose::Integer = 0,
  atol::T = T(sqrt(eps(T))),
  rtol::T = T(sqrt(eps(T))),
  itmax::Int = 0,
) where {T}
  return lsmr!(
    KS,
    A,
    ξ1,
    M = M,
    N = δ > zero(T) ? (one(T) / δ) * I : I,
    verbose = verbose,
    axtol = zero(T), # atol,
    btol = zero(T), # rtol,
    atol = atol,
    rtol = rtol,
    etol = zero(T),
    conlim = T(Inf),
    sqd = δ > zero(T),
    itmax = itmax,
  )
end

function ksolve!(
  KS::LnlqWorkspace{T},
  A,
  ξ2::AbstractVector{T},
  M,
  δ::T;
  verbose::Integer = 0,
  atol::T = T(sqrt(eps(T))),
  rtol::T = T(sqrt(eps(T))),
  itmax::Int = 0,
) where {T}
  return lnlq!(
    KS,
    A,
    ξ2,
    N = M,
    M = δ > zero(T) ? (one(T) / δ) * I : I,
    verbose = verbose,
    atol = atol,
    rtol = rtol,
    sqd = δ > zero(T),
    itmax = itmax,
  )
end

function ksolve!(
  KS::CraigWorkspace{T},
  A,
  ξ2::AbstractVector{T},
  M,
  δ::T;
  verbose::Integer = 0,
  atol::T = T(sqrt(eps(T))),
  rtol::T = T(sqrt(eps(T))),
  itmax::Int = 0,
) where {T}
  return craig!(
    KS,
    A,
    ξ2,
    N = M,
    M = δ > zero(T) ? (one(T) / δ) * I : I,
    λ = δ > zero(T) ? one(T) : zero(T),
    verbose = verbose,
    atol = atol,
    rtol = rtol,
    btol = zero(T),
    conlim = T(Inf),
    itmax = itmax,
  )
end

function ksolve!(
  KS::CraigmrWorkspace{T},
  A,
  ξ2::AbstractVector{T},
  M,
  δ::T;
  verbose::Integer = 0,
  atol::T = T(sqrt(eps(T))),
  rtol::T = T(sqrt(eps(T))),
  itmax::Int = 0,
) where {T}
  return craigmr!(
    KS,
    A,
    ξ2,
    N = M,
    M = δ > zero(T) ? (one(T) / δ) * I : I,
    verbose = verbose,
    atol = atol,
    rtol = rtol,
    sqd = true,
    itmax = itmax,
  )
end

function kscale!(rhs::AbstractVector{T}) where {T <: Real}
  rhsNorm = norm(rhs)
  if rhsNorm != zero(T)
    rhs ./= rhsNorm
  end
  return rhsNorm
end

function kunscale!(sol::AbstractVector{T}, rhsNorm::T) where {T <: Real}
  if rhsNorm != zero(T)
    sol .*= rhsNorm
  end
end

function update_kresiduals_history!(
  res::AbstractResiduals{T},
  K,
  sol::AbstractVector{T},
  rhs::AbstractVector{T},
) where {T <: Real}
  if typeof(res) <: ResidualsHistory
    mul!(res.KΔxy, K, sol) # krylov residuals
    res.Kres .= res.KΔxy .- rhs
  end
end

function update_kresiduals_history_K1struct!(
  res::AbstractResiduals{T},
  A,
  E,
  tmp,
  δ,
  sol::AbstractVector{T},
  rhs::AbstractVector{T},
  formul::Symbol,
) where {T <: Real}
  if typeof(res) <: ResidualsHistory
    mul!(tmp, A', sol)
    tmp ./= E
    @views mul!(res.KΔxy, A, tmp)
    if formul == :K1_1
      mul!(res.Kres, A, rhs ./ E, -one(T), zero(T))
    elseif formul == :K1_2
      res.Kres .= .-rhs
    end
    res.Kres .+= res.KΔxy .+ δ .* sol # residual computation
  end
end

get_krylov_method_name(KS::KrylovWorkspace) = uppercase(string(typeof(KS).name.name)[1:(end - 6)])

solver_name(pad::Union{PreallocatedDataNewtonKrylov, PreallocatedDataAugmentedKrylov}) = string(
  string(typeof(pad).name.name)[17:end],
  " with $(get_krylov_method_name(pad.KS))",
  " and $(precond_name(pad.pdat)) preconditioner",
)

solver_name(pad::PreallocatedDataNormalKrylov) =
  string(string(typeof(pad).name.name)[17:end], " with $(get_krylov_method_name(pad.KS))")

mutable struct GmresIRWorkspace{T, FC, S, Tr, Sr <: AbstractVector{Tr}} <: KrylovWorkspace{T, FC, S}
  solver::GmresWorkspace{T, FC, S}
  r::Sr
  rsolves::S
  x::S
  warm_start::Bool
  itertot::Int
end

Krylov.iteration_count(KS::GmresIRWorkspace) = KS.itertot

function Krylov.warm_start!(KS::GmresIRWorkspace, x) # only indicate to warm_start
  KS.warm_start = true
end

status_to_char(KS::GmresIRWorkspace) = status_to_char(KS.solver.stats.status)

function ksolve!(
  KS::GmresIRWorkspace{T},
  K,
  rhs::AbstractVector{T},
  P::LRPrecond;
  verbose::Integer = 0,
  atol::T = T(sqrt(eps(T))),
  rtol::T = T(sqrt(eps(T))),
  itmax::Int = 0,
) where {T}
  r = KS.r
  Tr = eltype(r)
  rsolves = KS.rsolves
  if KS.warm_start
    mul!(r, K, KS.x)
    @. r = rhs - r
  else
    r .= zero(Tr)
    KS.x .= zero(T)
  end
  KS.solver.warm_start = false
  iter = 0
  rsolves .= r
  optimal = iter ≥ itmax || norm(rsolves) ≤ atol
  while !optimal
    gmres!(
      KS.solver,
      K,
      rsolves,
      M = P.M,
      N = P.N,
      restart = false,
      verbose = verbose,
      atol = atol,
      rtol = rtol,
      itmax = length(KS.solver.c),
    )
    KS.x .+= KS.solver.x
    mul!(r, K, KS.x)
    @. r = rhs - r
    iter += Krylov.iteration_count(KS.solver)
    rsolves .= r
    optimal = iter ≥ itmax || Krylov.iteration_count(KS.solver) == 0 || norm(rsolves) ≤ atol
  end
  KS.itertot = iter
end

mutable struct IRWorkspace{T, S <: AbstractVector{T}, Tr, Sr <: AbstractVector{Tr}} <:
               KrylovWorkspace{T, T, S}
  x_solve1::S
  x_solve2::S
  r::Sr
  rsolves::S
  x::S
  warm_start::Bool
  itertot::Int
end

Krylov.iteration_count(KS::IRWorkspace) = KS.itertot

function Krylov.warm_start!(KS::IRWorkspace, x) # only indicate to warm_start
  KS.warm_start = true
end

status_to_char(KS::IRWorkspace) = 's'

function ksolve!(
  KS::IRWorkspace{T},
  K,
  rhs::AbstractVector{T},
  P::LRPrecond;
  verbose::Integer = 0,
  atol::T = T(sqrt(eps(T))),
  rtol::T = T(sqrt(eps(T))),
  itmax::Int = 0,
) where {T}
  r = KS.r
  Tr = eltype(r)
  rsolves = KS.rsolves
  if KS.warm_start
    mul!(r, K, KS.x)
    @. r = rhs - r
  else
    r .= zero(Tr)
    KS.x .= zero(T)
  end
  iter = 0
  rsolves .= r
  optimal = iter ≥ itmax || norm(rsolves) ≤ atol
  while !optimal
    mul!(KS.x_solve1, P.M, rsolves)
    mul!(KS.x_solve2, P.N, KS.x_solve1)
    KS.x .+= KS.x_solve2
    mul!(r, K, KS.x)
    @. r = rhs - r
    iter += 1
    rsolves .= r
    optimal = iter ≥ itmax || norm(rsolves) ≤ atol
  end
  KS.itertot = iter
end

function update_krylov_tol!(pad::PreallocatedData)
  if pad.atol > pad.atol_min
    pad.atol /= 10
  end
  if pad.rtol > pad.rtol_min
    pad.rtol /= 10
  end
end
