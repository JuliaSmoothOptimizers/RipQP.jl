# Tutorial

## Input

RipQP uses the package [QuadraticModels.jl](https://github.com/JuliaSmoothOptimizers/QuadraticModels.jl) to model
convex quadratic problems.

Here is a basic example:

```julia
using QuadraticModels
Q = [6. 2. 1.
    2. 5. 2.
    1. 2. 4.]
c = [-8.; -3; -3]
A = [1. 0. 1.
    0. 2. 1.]
b = [0.; 3]
l = [0.;0;0]
u = [Inf; Inf; Inf]
QM = QuadraticModel(c, Q, A=A, lcon=b, ucon=b, lvar=l, uvar=u, c0=0., name="QM")
```

It is also possible to use the package [QPSReader.jl](https://github.com/JuliaSmoothOptimizers/QPSReader.jl) in order to
read convex quadratic problems in MPS or SIF formats:

```julia
using QPSReader, QuadraticModels
QM = QuadraticModel(readqps("QAFIRO.SIF"))
```

## Solve the problem and read the statistics

Once your `QuadraticModel` is loaded, you can simply solve it with:

```julia
using RipQP
stats = ripqp(QM)
```

The `stats` output is a
[GenericExecutionStats](https://juliasmoothoptimizers.github.io/SolverCore.jl/dev/reference/#SolverCore.GenericExecutionStats)
from the package [SolverCore.jl](https://github.com/JuliaSmoothOptimizers/SolverCore.jl).

## Logging

RipQP displays some logs at each iterate.

You can deactivate logging with

```julia
stats = ripqp(QM, display = false)
```

## Change configuration and tolerances

The [`RipQP.InputConfig`](@ref) type allows the user to change the configuration of RipQP.
For example, you can use the multi-precision mode without scaling with:

```julia
stats = ripqp(QM, iconf = InputConfig(mode = :multi, scaling = false))
```

You can also change the [`RipQP.InputTol`](@ref) type to change the tolerances for the
stopping criteria:

```julia
stats = ripqp(QM, itol = InputTol(max_iter = 100, ϵ_rb = 1.0e-4),
              iconf = InputConfig(mode = :multi, scaling = false))
```

## Advanced: write your own solver

You can use your own solver to compute the direction of descent inside RipQP at each iteration.
Here is a basic example using the package [`LDLFactorizations.jl`](https://github.com/JuliaSmoothOptimizers/LDLFactorizations.jl).

First, you will need a [`RipQP.SolverParams`](@ref) to define parameters for your solver:

```julia
using RipQP, LinearAlgebra, LDLFactorizations, SparseArrays

struct K2basicLDLParams{T<:Real} <: SolverParams
    ρ :: T # dual regularization
    δ :: T # primal regularization
end
```

Then, you will have to create a type that allocates space for your solver, and a constructor using the following parameters:

```julia
mutable struct PreallocatedData_K2basic{T<:Real} <: RipQP.PreallocatedData{T}
    D                :: Vector{T} # temporary top-left diagonal of the K2 system
    ρ                :: T # dual regularization
    δ                :: T # primal regularization
    K                :: SparseMatrixCSC{T,Int} # K2 matrix
    K_fact           :: LDLFactorizations.LDLFactorization{T,Int,Int,Int} # factorized K2
end
```

Now you need to write a `RipQP.PreallocatedData` function that returns your type:

```julia
function RipQP.PreallocatedData(sp :: SolverParams, fd :: RipQP.QM_FloatData{T},
                                id :: RipQP.QM_IntData,
                                iconf :: InputConfig{Tconf}) where {T<:Real, Tconf<:Real}

    ρ, δ = T(sp.ρ), T(sp.δ)
    K = spzeros(T, id.ncon+id.nvar, id.ncon + id.nvar)
    K[1:id.nvar, 1:id.nvar] = .-fd.Q .- ρ .* Diagonal(ones(T, id.nvar))
    K[1:id.nvar, id.nvar+1:end] = fd.AT
    K[diagind(K)[id.nvar+1:end]] .= δ

    K_fact = ldl_analyze(Symmetric(K, :U))
    K_fact = ldl_factorize!(Symmetric(K, :U), K_fact)
    K_fact.__factorized = true

    return PreallocatedData_K2basic(zeros(T, id.nvar),
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
function RipQP.update_pad!(pad :: PreallocatedData_K2basic{T}, dda :: RipQP.DescentDirectionAllocs{T},
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
That is why the direction of descent `itd.Δxy` (resp. `dda.Δxy_aff` for the Predictor step)
countains the right hand side of the linear system to solve.

```julia
function RipQP.solver!(pad :: PreallocatedData_K2basic{T},
                       dda :: RipQP.DescentDirectionAllocsPC{T}, pt :: RipQP.Point{T},
                       itd :: RipQP.IterData{T}, fd :: RipQP.Abstract_QM_FloatData{T},
                       id :: RipQP.QM_IntData, res :: RipQP.Residuals{T},
                       cnts :: RipQP.Counters, T0 :: DataType,
                       step :: Symbol) where {T<:Real}

    if step == :aff # affine predictor step
        # solve the system and overwrite dda.Δxy_aff
        ldiv!(pad.K_fact, dda.Δxy_aff)
    else # for all other steps including the initial point
        # solve the system and overwrite itd.Δxy
        ldiv!(pad.K_fact, itd.Δxy)
    end

    return 0
end
```

Then, you can use your solver:

```julia
using QuadraticModels, QPSReader
qm = QuadraticModel(readqps("QAFIRO.SIF"))
stats1 = ripqp(qm, iconf = RipQP.InputConfig(sp = K2basicLDLParams(1.0e-6, 1.0e-6)))
```