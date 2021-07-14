abstract type PreallocatedData_K2Krylov{T <: Real, S} <: PreallocatedData{T, S} end
abstract type PreallocatedData_K2_5Krylov{T <: Real, S} <: PreallocatedData{T, S} end

function KSolver(s::Symbol)
  if s == :minres
    return :MinresSolver
  elseif s == :minres_qlp
    return :MinresQlpSolver
  end
end

mutable struct PreallocatedData_K2minres{T <: Real, S, Fv, Fu, Fw} <:
               PreallocatedData_K2Krylov{T, S}
  pdat::PreconditionerDataK2{T, S}
  D::S                                  # temporary top-left diagonal
  rhs::S
  regu::Regularization{T}
  δv::Vector{T}
  K::LinearOperator{T, Fv, Fu, Fw} # augmented matrix          
  KS::MinresSolver{T, S}
  atol::T
  rtol::T
end

mutable struct PreallocatedData_K2_5minres{T <: Real, S, Fv, Fu, Fw} <:
               PreallocatedData_K2_5Krylov{T, S}
  pdat::PreconditionerDataK2{T, S}
  D::S                                  # temporary top-left diagonal
  sqrtX1X2::S # vector to scale K2 to K2.5
  tmp::S # temporary vector for products
  rhs::S
  regu::Regularization{T}
  δv::Vector{T}
  K::LinearOperator{T, Fv, Fu, Fw} # augmented matrix          
  KS::MinresSolver{T, S}
  ratol::T
  rrtol::T
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

mutable struct PreallocatedData_K2minres_qlp{T <: Real, S, Fv, Fu, Fw} <:
               PreallocatedData_K2Krylov{T, S}
  pdat::PreconditionerDataK2{T, S}
  D::S                                  # temporary top-left diagonal
  rhs::S
  regu::Regularization{T}
  δv::Vector{T}
  K::LinearOperator{T, Fv, Fu, Fw} # augmented matrix          
  KS::MinresQlpSolver{T, S}
  atol::T
  rtol::T
end

mutable struct PreallocatedData_K2_5minres_qlp{T <: Real, S, Fv, Fu, Fw} <:
               PreallocatedData_K2_5Krylov{T, S}
  pdat::PreconditionerDataK2{T, S}
  D::S                                  # temporary top-left diagonal
  sqrtX1X2::S # vector to scale K2 to K2.5
  tmp::S # temporary vector for products
  rhs::S
  regu::Regularization{T}
  δv::Vector{T}
  K::LinearOperator{T, Fv, Fu, Fw} # augmented matrix          
  KS::MinresQlpSolver{T, S}
  ratol::T
  rrtol::T
end

ksolve!(
  KS::MinresQlpSolver{T, S},
  K,
  rhs::AbstractVector{T},
  M;
  verbose::Integer = 0,
  atol::T = T(sqrt(eps(T))),
  rtol::T = T(sqrt(eps(T))),
) where {T, S} = minres_qlp!(KS, K, rhs, M = M, verbose = verbose, atol = atol, rtol = rtol)
