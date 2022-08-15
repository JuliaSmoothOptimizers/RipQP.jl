export AbstractFactorization, LDLFact, HSLMA57Fact

abstract type AbstractFactorization end

struct LDLFact <: AbstractFactorization
  regul::Symbol
  function LDLFact(regul::Symbol)
    regul == :classic || regul == :dynamic ||
      regul == :hybrid ||
      regul == :none ||
      error("regul should be :classic or :dynamic or :hybrid or :none")
    return new(regul)
  end
end

LDLFact(; regul::Symbol = :classic) = LDLFact(regul)

include("ldlfact_utils.jl")

struct HSLMA57Fact <: AbstractFactorization
  regul::Symbol
  function HSLMA57Fact(regul::Symbol)
    regul == :classic || regul == :none ||
      error("regul should be :classic or :dynamic or :hybrid or :none")
    return new(regul)
  end
end

HSLMA57Fact(; regul::Symbol = :classic) = HSLMA57Fact(regul)
