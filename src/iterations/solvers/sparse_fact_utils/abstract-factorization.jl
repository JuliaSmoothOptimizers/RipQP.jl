export AbstractFactorization, LDLFact, HSLMA57Fact, HSLMA97Fact, CholmodFact, QDLDLFact

import LinearAlgebra.ldiv!

"""
Abstract type to select a factorization algorithm.
"""
abstract type AbstractFactorization end

abstract type FactorizationData{T} end

function ldiv!(res::AbstractVector, K_fact::FactorizationData, v::AbstractVector)
  res .= v
  ldiv!(K_fact, res)
end

"""
    fact_alg = LDLFact(; regul = :classic)

Choose [`LDLFactorizations.jl`](https://github.com/JuliaSmoothOptimizers/LDLFactorizations.jl) to compute factorizations.
"""
struct LDLFact <: AbstractFactorization
  regul::Symbol
  function LDLFact(regul::Symbol)
    regul == :classic ||
      regul == :dynamic ||
      regul == :hybrid ||
      regul == :none ||
      error("regul should be :classic or :dynamic or :hybrid or :none")
    return new(regul)
  end
end
LDLFact(; regul::Symbol = :classic) = LDLFact(regul)
include("ldlfact_utils.jl")

"""
    fact_alg = CholmodFact(; regul = :classic)

Choose `ldlt` from Cholmod to compute factorizations.
`using SuiteSparse` should be used before `using RipQP`.
"""
struct CholmodFact <: AbstractFactorization
  regul::Symbol
  function CholmodFact(regul::Symbol)
    regul == :classic || regul == :none || error("regul should be :classic or :none")
    return new(regul)
  end
end
CholmodFact(; regul::Symbol = :classic) = CholmodFact(regul)

"""
    fact_alg = QDLDLFact(; regul = :classic)

Choose [`QDLDL.jl`](https://github.com/oxfordcontrol/QDLDL.jl) to compute factorizations.
`using QDLDL` should be used before `using RipQP`.
"""
struct QDLDLFact <: AbstractFactorization
  regul::Symbol
  function QDLDLFact(regul::Symbol)
    regul == :classic || regul == :none || error("regul should be :classic or :none")
    return new(regul)
  end
end
QDLDLFact(; regul::Symbol = :classic) = QDLDLFact(regul)

"""
    fact_alg = HSLMA57Fact(; regul = :classic)

Choose [`HSL.jl`](https://github.com/JuliaSmoothOptimizers/HSL.jl) MA57 to compute factorizations.
`using HSL` should be used before `using RipQP`.
"""
struct HSLMA57Fact <: AbstractFactorization
  regul::Symbol
  sqd::Bool
  function HSLMA57Fact(regul::Symbol, sqd::Bool)
    regul == :classic || regul == :none || error("regul should be :classic or :none")
    return new(regul, sqd)
  end
end
HSLMA57Fact(; regul::Symbol = :classic, sqd::Bool = true) = HSLMA57Fact(regul, sqd)

"""
    fact_alg = HSLMA97Fact(; regul = :classic)

Choose [`HSL.jl`](https://github.com/JuliaSmoothOptimizers/HSL.jl) MA57 to compute factorizations.
`using HSL` should be used before `using RipQP`.
"""
struct HSLMA97Fact <: AbstractFactorization
  regul::Symbol
  sqd::Bool
  function HSLMA97Fact(regul::Symbol, sqd::Bool)
    regul == :classic || regul == :none || error("regul should be :classic or :none")
    return new(regul, sqd)
  end
end
HSLMA97Fact(; regul::Symbol = :classic, sqd::Bool = true) = HSLMA57Fact(regul, sqd)
