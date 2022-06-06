# Switching solvers

## Solve method

By default, RipQP uses a predictor-corrector algorithm that solves two linear systems per interior-point iteration.
This is efficient when using a factorization to solve the interior-point system.
It is also possible to use an infeasible path-following algorithm, which is efficient when solving each system is more expensive (for example when using a Krylov method without preconditioner).

```julia
stats = ripqp(qm, solve_method = PC()) # predictor-corrector (default)
stats = ripqp(qm, solve_method = IPF()) # infeasible path-following
```

## Choosing a solver

It is possible to choose different solvers to solve the interior-point system.
All these solvers can be called with a structure that is a subtype of a [`RipQP.SolverParams`](@ref).
The default solver is [`RipQP.K2LDLParams`](@ref).

There are a lot of solvers that are implemented using Krylov methods from [`Krylov.jl`](https://github.com/JuliaSmoothOptimizers/Krylov.jl).
For example:

```julia
stats = ripqp(qm, sp = K2KrylovParams())
```

These solvers are usually less efficient, but they can be used with a preconditioner to improve the performances.
All these preconditioners can be called with a structure that is a subtype of an [`AbstractPreconditioner`](@ref).
By default, most Krylov solvers use the [`RipQP.Identity`](@ref) preconditioner, but is possible to use for example
a LDL factorization [`RipQP.LDL`](@ref).
The Krylov method then acts as form of using iterative refinement to a LDL factorization of K2:

```julia
stats = ripqp(qm, sp = K2KrylovParams(uplo = :U, preconditioner = LDL())) 
# uplo = :U is mandatory with this preconditioner
```

It is also possible to change the Krylov method used to solve the system:
```julia
stats = ripqp(qm, sp = K2KrylovParams(uplo = :U, kmethod = :gmres, preconditioner = LDL()))
```

## Advanced: write your own solver

You can use your own solver to compute the direction of descent inside RipQP at each iteration.
Here is a basic example using the package [`LDLFactorizations.jl`](https://github.com/JuliaSmoothOptimizers/LDLFactorizations.jl).

First, you will need a [`RipQP.SolverParams`](@ref) to define parameters for your solver:

```julia
using RipQP, LinearAlgebra, LDLFactorizations, SparseArrays

struct K2basicLDLParams{T<:Real} <: SolverParams
    uplo   :: Symbol # mandatory, tells RipQP which triangle of the augmented system to store
    ρ      :: T # dual regularization
    δ      :: T # primal regularization
end
```

Then, you will have to create a type that allocates space for your solver, and a constructor using the following parameters:

```julia
mutable struct PreallocatedDataK2basic{T<:Real, S} <: RipQP.PreallocatedDataAugmented{T, S}
    D                :: S # temporary top-left diagonal of the K2 system
    ρ                :: T # dual regularization
    δ                :: T # primal regularization
    K                :: SparseMatrixCSC{T,Int} # K2 matrix
    K_fact           :: LDLFactorizations.LDLFactorization{T,Int,Int,Int} # factorized K2
end
```

Now you need to write a `RipQP.PreallocatedData` function that returns your type:

```julia
function RipQP.PreallocatedData(sp :: SolverParams, fd :: RipQP.QM_FloatData{T},
                                id :: RipQP.QM_IntData, itd :: RipQP.IterData{T},
                                pt :: RipQP.Point{T},
                                iconf :: InputConfig{Tconf}) where {T<:Real, Tconf<:Real}

    ρ, δ = T(sp.ρ), T(sp.δ)
    K = spzeros(T, id.ncon+id.nvar, id.ncon + id.nvar)
    K[1:id.nvar, 1:id.nvar] = .-fd.Q .- ρ .* Diagonal(ones(T, id.nvar))
    # A = Aᵀ of the input QuadraticModel since we use the upper triangle:
    K[1:id.nvar, id.nvar+1:end] = fd.A 
    K[diagind(K)[id.nvar+1:end]] .= δ

    K_fact = ldl_analyze(Symmetric(K, :U))
    @assert sp.uplo == :U # LDLFactorizations does not work with the lower triangle
    K_fact = ldl_factorize!(Symmetric(K, :U), K_fact)
    K_fact.__factorized = true

    return PreallocatedDataK2basic(zeros(T, id.nvar),
                                    ρ,
                                    δ,
                                    K, #K
                                    K_fact #K_fact
                                    )
end
```

Then, you need to write a `RipQP.update_pad!` function that will update the `RipQP.PreallocatedData`
struct before computing the direction of descent.

```julia
function RipQP.update_pad!(pad :: PreallocatedDataK2basic{T}, dda :: RipQP.DescentDirectionAllocs{T},
                           pt :: RipQP.Point{T}, itd :: RipQP.IterData{T},
                           fd :: RipQP.Abstract_QM_FloatData{T}, id :: RipQP.QM_IntData,
                           res :: RipQP.Residuals{T}, cnts :: RipQP.Counters,
                           T0 :: RipQP.DataType) where {T<:Real}

    # update the diagonal of K2
    pad.D .= -pad.ρ
    pad.D[id.ilow] .-= pt.s_l ./ itd.x_m_lvar
    pad.D[id.iupp] .-= pt.s_u ./ itd.uvar_m_x
    pad.D .-= fd.Q[diagind(fd.Q)]
    pad.K[diagind(pad.K)[1:id.nvar]] = pad.D
    pad.K[diagind(pad.K)[id.nvar+1:end]] .= pad.δ

    # factorize K2
    ldl_factorize!(Symmetric(pad.K, :U), pad.K_fact)

end
```

Finally, you need to write a `RipQP.solver!` function that compute directions of descent.
Note that this function solves in-place the linear system by overwriting the direction of descent.
That is why the direction of descent `dd` 
countains the right hand side of the linear system to solve.

```julia
function RipQP.solver!(dd :: AbstractVector{T}, pad :: PreallocatedDataK2basic{T},
                       dda :: RipQP.DescentDirectionAllocsPC{T}, pt :: RipQP.Point{T},
                       itd :: RipQP.IterData{T}, fd :: RipQP.Abstract_QM_FloatData{T},
                       id :: RipQP.QM_IntData, res :: RipQP.Residuals{T},
                       cnts :: RipQP.Counters, T0 :: DataType,
                       step :: Symbol) where {T<:Real}

    ldiv!(pad.K_fact, dd)
    return 0
end
```

Then, you can use your solver:

```julia
using QuadraticModels, QPSReader
qm = QuadraticModel(readqps("QAFIRO.SIF"))
stats1 = ripqp(qm, sp = K2basicLDLParams(:U, 1.0e-6, 1.0e-6))
```
