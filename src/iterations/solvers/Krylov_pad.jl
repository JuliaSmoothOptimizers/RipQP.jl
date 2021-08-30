abstract type PreallocatedData_Krylov{T <: Real, S} <: PreallocatedData{T, S} end

function KSolver(s::Symbol)
  if s == :minres
    return :MinresSolver
  elseif s == :minres_qlp
    return :MinresQlpSolver
  end
end

mutable struct PreallocatedData_K2Krylov{T <: Real, S, L <: LinearOperator, 
               Pr <: PreconditionerDataK2, Ksol <: KrylovSolver} <: PreallocatedData_Krylov{T, S}
  pdat::Pr
  D::S                                  # temporary top-left diagonal
  rhs::S
  regu::Regularization{T}
  δv::Vector{T}
  K::L # augmented matrix (LinearOperator)         
  KS::Ksol
  atol::T
  rtol::T
  atol_min::T
  rtol_min::T
end

mutable struct PreallocatedData_K2_5Krylov{T <: Real, S, L <: LinearOperator, Pr <: PreconditionerDataK2,
               Ksol <: KrylovSolver} <: PreallocatedData_Krylov{T, S}
  pdat::Pr
  D::S                                  # temporary top-left diagonal
  sqrtX1X2::S # vector to scale K2 to K2.5
  tmp::S # temporary vector for products
  rhs::S
  regu::Regularization{T}
  δv::Vector{T}
  K::L # augmented matrix          
  KS::Ksol
  atol::T
  rtol::T
  atol_min::T
  rtol_min::T
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
