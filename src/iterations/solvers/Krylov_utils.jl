function KSolver(s::Symbol)
  if s == :minres
    return :MinresSolver
  elseif s == :minres_qlp
    return :MinresQlpSolver
  elseif s == :symmlq
    return :SymmlqSolver
  elseif s == :cg
    return :CgSolver
  elseif s == :cg_lanczos
    return :CgLanczosSolver
  elseif s == :cr
    return :CrSolver
  elseif s == :bilq
    return :BilqSolver
  elseif s == :qmr
    return :QmrSolver
  elseif s == :usymlq
    return :UsymlqSolver
  elseif s == :usymqr
    return :UsymqrSolver
  elseif s == :bicgstab
    return :BicgstabSolver
  elseif s == :diom
    return :DiomSolver
  elseif s == :fom
    return :FomSolver
  elseif s == :dqgmres
    return :DqgmresSolver
  elseif s == :gmres
    return :GmresSolver
  elseif s == :tricg
    return :TricgSolver
  elseif s == :trimr
    return :TrimrSolver
  elseif s == :gpmr
    return :GpmrSolver
  elseif s == :lsqr
    return :LsqrSolver
  elseif s == :lsmr
    return :LsmrSolver
  end
end

function init_Ksolver(M, v, sp::SolverParams)
  kmethod = sp.kmethod
  if kmethod ∈ [:gpmr, :diom, :fom, :gmres, :dqgmres]
    return eval(KSolver(kmethod))(M, v, sp.mem)
  end
  return eval(KSolver(kmethod))(M, v)
end

ksolve!(
  KS::MinresSolver{T, S},
  K,
  rhs::AbstractVector{T},
  M;
  verbose::Integer = 0,
  atol::T = T(sqrt(eps(T))),
  rtol::T = T(sqrt(eps(T))),
) where {T, S} = minres!(
  KS,
  K,
  rhs,
  M = M,
  verbose = verbose,
  atol = zero(T),
  rtol = zero(T),
  ratol = atol,
  rrtol = rtol,
)

ksolve!(
  KS::MinresQlpSolver{T, S},
  K,
  rhs::AbstractVector{T},
  M;
  verbose::Integer = 0,
  atol::T = T(sqrt(eps(T))),
  rtol::T = T(sqrt(eps(T))),
) where {T, S} = minres_qlp!(KS, K, rhs, M = M, verbose = verbose, atol = atol, rtol = rtol)

ksolve!(
  KS::SymmlqSolver{T, S},
  K,
  rhs::AbstractVector{T},
  M;
  verbose::Integer = 0,
  atol::T = T(sqrt(eps(T))),
  rtol::T = T(sqrt(eps(T))),
) where {T, S} = symmlq!(KS, K, rhs, M = M, verbose = verbose, atol = atol, rtol = rtol)

ksolve!(
  KS::CgSolver{T, S},
  K,
  rhs::AbstractVector{T},
  M;
  verbose::Integer = 0,
  atol::T = T(sqrt(eps(T))),
  rtol::T = T(sqrt(eps(T))),
) where {T, S} = cg!(KS, K, rhs, M = M, verbose = verbose, atol = atol, rtol = rtol)

ksolve!(
  KS::CgLanczosSolver{T, S},
  K,
  rhs::AbstractVector{T},
  M;
  verbose::Integer = 0,
  atol::T = T(sqrt(eps(T))),
  rtol::T = T(sqrt(eps(T))),
) where {T, S} = cg_lanczos!(KS, K, rhs, M = M, verbose = verbose, atol = atol, rtol = rtol)

ksolve!(
  KS::CrSolver{T, S},
  K,
  rhs::AbstractVector{T},
  M;
  verbose::Integer = 0,
  atol::T = T(sqrt(eps(T))),
  rtol::T = T(sqrt(eps(T))),
) where {T, S} = cr!(KS, K, rhs, M = M, verbose = verbose, atol = atol, rtol = rtol)

ksolve!(
  KS::BilqSolver{T, S},
  K,
  rhs::AbstractVector{T},
  M;
  verbose::Integer = 0,
  atol::T = T(sqrt(eps(T))),
  rtol::T = T(sqrt(eps(T))),
) where {T, S} = bilq!(KS, K, rhs, verbose = verbose, atol = atol, rtol = rtol)

ksolve!(
  KS::QmrSolver{T, S},
  K,
  rhs::AbstractVector{T},
  M;
  verbose::Integer = 0,
  atol::T = T(sqrt(eps(T))),
  rtol::T = T(sqrt(eps(T))),
) where {T, S} = qmr!(KS, K, rhs, verbose = verbose, atol = atol, rtol = rtol)

ksolve!(
  KS::UsymlqSolver{T, S},
  K,
  rhs::AbstractVector{T},
  M;
  verbose::Integer = 0,
  atol::T = T(sqrt(eps(T))),
  rtol::T = T(sqrt(eps(T))),
) where {T, S} = usymlq!(KS, K, rhs, rhs, verbose = verbose, atol = atol, rtol = rtol)

ksolve!(
  KS::UsymqrSolver{T, S},
  K,
  rhs::AbstractVector{T},
  M;
  verbose::Integer = 0,
  atol::T = T(sqrt(eps(T))),
  rtol::T = T(sqrt(eps(T))),
) where {T, S} = usymqr!(KS, K, rhs, rhs, verbose = verbose, atol = atol, rtol = rtol)

ksolve!(
  KS::BicgstabSolver{T, S},
  K,
  rhs::AbstractVector{T},
  M;
  verbose::Integer = 0,
  atol::T = T(sqrt(eps(T))),
  rtol::T = T(sqrt(eps(T))),
) where {T, S} = bicgstab!(KS, K, rhs, verbose = verbose, atol = atol, rtol = rtol)

ksolve!(
  KS::DiomSolver{T, S},
  K,
  rhs::AbstractVector{T},
  M;
  verbose::Integer = 0,
  atol::T = T(sqrt(eps(T))),
  rtol::T = T(sqrt(eps(T))),
) where {T, S} = diom!(KS, K, rhs, verbose = verbose, atol = atol, rtol = rtol)

ksolve!(
  KS::FomSolver{T, S},
  K,
  rhs::AbstractVector{T},
  M;
  verbose::Integer = 0,
  atol::T = T(sqrt(eps(T))),
  rtol::T = T(sqrt(eps(T))),
) where {T, S} = fom!(KS, K, rhs, verbose = verbose, atol = atol, rtol = rtol)

ksolve!(
  KS::DqgmresSolver{T, S},
  K,
  rhs::AbstractVector{T},
  M;
  verbose::Integer = 0,
  atol::T = T(sqrt(eps(T))),
  rtol::T = T(sqrt(eps(T))),
) where {T, S} = dqgmres!(KS, K, rhs, verbose = verbose, atol = atol, rtol = rtol)

ksolve!(
  KS::GmresSolver{T, S},
  K,
  rhs::AbstractVector{T},
  M;
  verbose::Integer = 0,
  atol::T = T(sqrt(eps(T))),
  rtol::T = T(sqrt(eps(T))),
) where {T, S} = gmres!(KS, K, rhs, verbose = verbose, atol = atol, rtol = rtol)

ksolve!(
  KS::TricgSolver{T, S},
  A,
  ξ1::AbstractVector{T},
  ξ2::AbstractVector{T},
  M,
  N;
  verbose::Integer = 0,
  atol::T = T(sqrt(eps(T))),
  rtol::T = T(sqrt(eps(T))),
  gsp::Bool = false,
) where {T, S} =
  tricg!(KS, A, ξ1, ξ2, M = M, N = N, flip = true, verbose = verbose, atol = atol, rtol = rtol)

ksolve!(
  KS::TrimrSolver{T, S},
  A,
  ξ1::AbstractVector{T},
  ξ2::AbstractVector{T},
  M,
  N;
  verbose::Integer = 0,
  atol::T = T(sqrt(eps(T))),
  rtol::T = T(sqrt(eps(T))),
  gsp::Bool = false,
) where {T, S} = trimr!(
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
)

function ksolve!(
  KS::GpmrSolver{T, S},
  A,
  ξ1::AbstractVector{T},
  ξ2::AbstractVector{T},
  M,
  N;
  verbose::Integer = 0,
  atol::T = T(sqrt(eps(T))),
  rtol::T = T(sqrt(eps(T))),
  gsp::Bool = false,
) where {T, S}
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
  )
end

# gpmr solver for K3.5
function ksolve!(
  KS::GpmrSolver{T, S},
  A,
  ξ1::AbstractVector{T},
  ξ2::AbstractVector{T},
  M,
  N::AbstractLinearOperator{T};
  verbose::Integer = 0,
  atol::T = T(sqrt(eps(T))),
  rtol::T = T(sqrt(eps(T))),
) where {T, S}
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
  )
end

function ksolve!(
  KS::LsqrSolver{T, S},
  A,
  ξ1::S,
  M,
  δ::T;
  verbose::Integer = 0,
  atol::T = T(sqrt(eps(T))),
  rtol::T = T(sqrt(eps(T))),
) where {T, S}
  return lsqr!(
    KS,
    A,
    ξ1,
    M = M,
    N = one(T)/δ * I,
    # λ = sqrt(δ),
    verbose = verbose,
    axtol = atol,
    btol = rtol,
    sqd = true,
  )
end

function ksolve!(
  KS::LsmrSolver{T, S},
  A,
  ξ1::S,
  M,
  δ::T;
  verbose::Integer = 0,
  atol::T = T(sqrt(eps(T))),
  rtol::T = T(sqrt(eps(T))),
) where {T, S}
  return lsmr!(
    KS,
    A,
    ξ1,
    M = M,
    # N = one(T)/δ * I,
    λ = sqrt(δ),
    verbose = verbose,
    axtol = atol,
    btol = rtol,
    sqd = false,
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
    res.Kres = res.KΔxy .- rhs
  end
end
