get_uplo(fact_alg::LDLFact) = :U

mutable struct LDLFactorizationData{T} <: FactorizationData{T}
  LDL::LDLFactorizations.LDLFactorization{T, Int, Int, Int}
end

init_fact(K::Symmetric{T, SparseMatrixCSC{T, Int}}, fact_alg::LDLFact; Tf = T) where {T} =
  LDLFactorizationData(ldl_analyze(K, Tf = Tf))

generic_factorize!(K::Symmetric, K_fact::LDLFactorizationData) =
  ldl_factorize!(K, K_fact.LDL)

LDLFactorizations.factorized(K_fact::LDLFactorizationData) = factorized(K_fact.LDL)

ldiv!(K_fact::LDLFactorizationData, dd::AbstractVector) = ldiv!(K_fact.LDL, dd)

# used only for preconditioner
function abs_diagonal!(K_fact::LDLFactorizationData)
  K_fact.LDL.d .= abs.(K_fact.LDL.d)
end

  # LDLFactorization conversion function
convertldl(T::DataType, K_fact::LDLFactorizationData) = LDLFactorizationData(
    LDLFactorizations.LDLFactorization(
    K_fact.LDL.__analyzed,
    K_fact.LDL.__factorized,
    K_fact.LDL.__upper,
    K_fact.LDL.n,
    K_fact.LDL.parent,
    K_fact.LDL.Lnz,
    K_fact.LDL.flag,
    K_fact.LDL.P,
    K_fact.LDL.pinv,
    K_fact.LDL.Lp,
    K_fact.LDL.Cp,
    K_fact.LDL.Ci,
    K_fact.LDL.Li,
    convert(Array{T}, K_fact.LDL.Lx),
    convert(Array{T}, K_fact.LDL.d),
    convert(Array{T}, K_fact.LDL.Y),
    K_fact.LDL.pattern,
    T(K_fact.LDL.r1),
    T(K_fact.LDL.r2),
    T(K_fact.LDL.tol),
    K_fact.LDL.n_d,
  ),
)