function KSolver(s::Symbol)
  if s == :minres
    return :MinresSolver
  elseif s == :minres_qlp
    return :MinresQlpSolver
  elseif s == :qmr
    return :QmrSolver
  elseif s == :usymqr
    return :UsymqrSolver
  elseif s == :bicgstab
    return :BicgstabSolver
  end
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
  KS::QmrSolver{T, S},
  K,
  rhs::AbstractVector{T},
  M;
  verbose::Integer = 0,
  atol::T = T(sqrt(eps(T))),
  rtol::T = T(sqrt(eps(T))),
) where {T, S} = qmr!(KS, K, rhs, verbose = verbose, atol = atol, rtol = rtol)

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