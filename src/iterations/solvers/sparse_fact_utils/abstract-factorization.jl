export AbstractFactorization, LDLFact, HSLMA57Fact, CholmodFact, QDLDLFact

import LinearAlgebra.ldiv!

abstract type AbstractFactorization end

abstract type FactorizationData{T} end

function ldiv!(res::AbstractVector, K_fact::FactorizationData, v::AbstractVector)
  res .= v
  ldiv!(K_fact, res)
end

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

struct CholmodFact <: AbstractFactorization
  regul::Symbol
  function CholmodFact(regul::Symbol)
    regul == :classic || regul == :none || error("regul should be :classic or :none")
    return new(regul)
  end
end
CholmodFact(; regul::Symbol = :classic) = CholmodFact(regul)

struct QDLDLFact <: AbstractFactorization
  regul::Symbol
  function QDLDLFact(regul::Symbol)
    regul == :classic || regul == :none || error("regul should be :classic or :none")
    return new(regul)
  end
end
QDLDLFact(; regul::Symbol = :classic) = QDLDLFact(regul)

struct HSLMA57Fact <: AbstractFactorization
  regul::Symbol
  sqd::Bool
  function HSLMA57Fact(regul::Symbol, sqd::Bool)
    regul == :classic || regul == :none || error("regul should be :classic or :none")
    return new(regul, sqd)
  end
end
HSLMA57Fact(; regul::Symbol = :classic, sqd::Bool = true) = HSLMA57Fact(regul, sqd)
