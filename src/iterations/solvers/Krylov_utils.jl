import Krylov.KRYLOV_SOLVERS

function init_Ksolver(M, v, sp::SolverParams)
  kmethod = sp.kmethod
  if kmethod ∈ (:gpmr, :diom, :fom, :dqgmres, :gmres)
    return eval(KRYLOV_SOLVERS[kmethod])(M, v, sp.mem)
  end
  return eval(KRYLOV_SOLVERS[kmethod])(M, v)
end

ksolve!(
  KS::MinresSolver{T},
  K,
  rhs::AbstractVector{T},
  M;
  verbose::Integer = 0,
  atol::T = T(sqrt(eps(T))),
  rtol::T = T(sqrt(eps(T))),
) where {T} = minres!(
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
  KS::MinresQlpSolver{T},
  K,
  rhs::AbstractVector{T},
  M;
  verbose::Integer = 0,
  atol::T = T(sqrt(eps(T))),
  rtol::T = T(sqrt(eps(T))),
) where {T} = minres_qlp!(KS, K, rhs, M = M, verbose = verbose, atol = atol, rtol = rtol)

ksolve!(
  KS::SymmlqSolver{T},
  K,
  rhs::AbstractVector{T},
  M;
  verbose::Integer = 0,
  atol::T = T(sqrt(eps(T))),
  rtol::T = T(sqrt(eps(T))),
) where {T} = symmlq!(KS, K, rhs, M = M, verbose = verbose, atol = atol, rtol = rtol)

ksolve!(
  KS::CgSolver{T},
  K,
  rhs::AbstractVector{T},
  M;
  verbose::Integer = 0,
  atol::T = T(sqrt(eps(T))),
  rtol::T = T(sqrt(eps(T))),
) where {T} = cg!(KS, K, rhs, M = M, verbose = verbose, atol = atol, rtol = rtol)

ksolve!(
  KS::CgLanczosSolver{T},
  K,
  rhs::AbstractVector{T},
  M;
  verbose::Integer = 0,
  atol::T = T(sqrt(eps(T))),
  rtol::T = T(sqrt(eps(T))),
) where {T} = cg_lanczos!(KS, K, rhs, M = M, verbose = verbose, atol = atol, rtol = rtol)

ksolve!(
  KS::CrSolver{T},
  K,
  rhs::AbstractVector{T},
  M;
  verbose::Integer = 0,
  atol::T = T(sqrt(eps(T))),
  rtol::T = T(sqrt(eps(T))),
) where {T} = cr!(KS, K, rhs, M = M, verbose = verbose, atol = atol, rtol = rtol)

ksolve!(
  KS::BilqSolver{T},
  K,
  rhs::AbstractVector{T},
  M;
  verbose::Integer = 0,
  atol::T = T(sqrt(eps(T))),
  rtol::T = T(sqrt(eps(T))),
) where {T} = bilq!(KS, K, rhs, verbose = verbose, atol = atol, rtol = rtol)

ksolve!(
  KS::QmrSolver{T},
  K,
  rhs::AbstractVector{T},
  M;
  verbose::Integer = 0,
  atol::T = T(sqrt(eps(T))),
  rtol::T = T(sqrt(eps(T))),
) where {T} = qmr!(KS, K, rhs, verbose = verbose, atol = atol, rtol = rtol)

ksolve!(
  KS::UsymlqSolver{T},
  K,
  rhs::AbstractVector{T},
  M;
  verbose::Integer = 0,
  atol::T = T(sqrt(eps(T))),
  rtol::T = T(sqrt(eps(T))),
) where {T} = usymlq!(KS, K, rhs, rhs, verbose = verbose, atol = atol, rtol = rtol)

ksolve!(
  KS::UsymqrSolver{T},
  K,
  rhs::AbstractVector{T},
  M;
  verbose::Integer = 0,
  atol::T = T(sqrt(eps(T))),
  rtol::T = T(sqrt(eps(T))),
) where {T} = usymqr!(KS, K, rhs, rhs, verbose = verbose, atol = atol, rtol = rtol)

ksolve!(
  KS::BicgstabSolver{T},
  K,
  rhs::AbstractVector{T},
  M;
  verbose::Integer = 0,
  atol::T = T(sqrt(eps(T))),
  rtol::T = T(sqrt(eps(T))),
) where {T} = bicgstab!(KS, K, rhs, verbose = verbose, atol = atol, rtol = rtol)

ksolve!(
  KS::DiomSolver{T},
  K,
  rhs::AbstractVector{T},
  M;
  verbose::Integer = 0,
  atol::T = T(sqrt(eps(T))),
  rtol::T = T(sqrt(eps(T))),
) where {T} = diom!(KS, K, rhs, verbose = verbose, atol = atol, rtol = rtol)

ksolve!(
  KS::FomSolver{T},
  K,
  rhs::AbstractVector{T},
  M;
  verbose::Integer = 0,
  atol::T = T(sqrt(eps(T))),
  rtol::T = T(sqrt(eps(T))),
) where {T} = fom!(KS, K, rhs, verbose = verbose, atol = atol, rtol = rtol)

ksolve!(
  KS::DqgmresSolver{T},
  K,
  rhs::AbstractVector{T},
  M;
  verbose::Integer = 0,
  atol::T = T(sqrt(eps(T))),
  rtol::T = T(sqrt(eps(T))),
) where {T} = dqgmres!(KS, K, rhs, verbose = verbose, atol = atol, rtol = rtol)

ksolve!(
  KS::GmresSolver{T},
  K,
  rhs::AbstractVector{T},
  M;
  verbose::Integer = 0,
  atol::T = T(sqrt(eps(T))),
  rtol::T = T(sqrt(eps(T))),
) where {T} = gmres!(KS, K, rhs, verbose = verbose, atol = atol, rtol = rtol)

ksolve!(
  KS::TricgSolver{T},
  A,
  ξ1::AbstractVector{T},
  ξ2::AbstractVector{T},
  M,
  N;
  verbose::Integer = 0,
  atol::T = T(sqrt(eps(T))),
  rtol::T = T(sqrt(eps(T))),
  gsp::Bool = false,
) where {T} =
  tricg!(KS, A, ξ1, ξ2, M = M, N = N, flip = true, verbose = verbose, atol = atol, rtol = rtol)

ksolve!(
  KS::TrimrSolver{T},
  A,
  ξ1::AbstractVector{T},
  ξ2::AbstractVector{T},
  M,
  N;
  verbose::Integer = 0,
  atol::T = T(sqrt(eps(T))),
  rtol::T = T(sqrt(eps(T))),
  gsp::Bool = false,
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
)

function ksolve!(
  KS::GpmrSolver{T},
  A,
  ξ1::AbstractVector{T},
  ξ2::AbstractVector{T},
  M,
  N;
  verbose::Integer = 0,
  atol::T = T(sqrt(eps(T))),
  rtol::T = T(sqrt(eps(T))),
  gsp::Bool = false,
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
  )
end

# gpmr solver for K3.5
function ksolve!(
  KS::GpmrSolver{T},
  A,
  ξ1::AbstractVector{T},
  ξ2::AbstractVector{T},
  M,
  N::AbstractLinearOperator{T};
  verbose::Integer = 0,
  atol::T = T(sqrt(eps(T))),
  rtol::T = T(sqrt(eps(T))),
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
  )
end

function ksolve!(
  KS::LslqSolver{T},
  A,
  ξ1::AbstractVector{T},
  M,
  δ::T;
  verbose::Integer = 0,
  atol::T = T(sqrt(eps(T))),
  rtol::T = T(sqrt(eps(T))),
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
  )
end

function ksolve!(
  KS::LsqrSolver{T},
  A,
  ξ1::AbstractVector{T},
  M,
  δ::T;
  verbose::Integer = 0,
  atol::T = T(sqrt(eps(T))),
  rtol::T = T(sqrt(eps(T))),
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
  )
end

function ksolve!(
  KS::LsmrSolver{T},
  A,
  ξ1::AbstractVector{T},
  M,
  δ::T;
  verbose::Integer = 0,
  atol::T = T(sqrt(eps(T))),
  rtol::T = T(sqrt(eps(T))),
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
  )
end

function ksolve!(
  KS::LnlqSolver{T},
  A,
  ξ2::AbstractVector{T},
  M,
  δ::T;
  verbose::Integer = 0,
  atol::T = T(sqrt(eps(T))),
  rtol::T = T(sqrt(eps(T))),
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
  )
end

function ksolve!(
  KS::CraigSolver{T},
  A,
  ξ2::AbstractVector{T},
  M,
  δ::T;
  verbose::Integer = 0,
  atol::T = T(sqrt(eps(T))),
  rtol::T = T(sqrt(eps(T))),
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
  )
end

function ksolve!(
  KS::CraigmrSolver{T},
  A,
  ξ2::AbstractVector{T},
  M,
  δ::T;
  verbose::Integer = 0,
  atol::T = T(sqrt(eps(T))),
  rtol::T = T(sqrt(eps(T))),
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
