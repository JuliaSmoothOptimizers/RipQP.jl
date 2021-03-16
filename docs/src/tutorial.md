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
[GenericExecutionStats](https://juliasmoothoptimizers.github.io/SolverTools.jl/stable/api/#SolverTools.GenericExecutionStats) 
from the package [SolverTools.jl](https://github.com/JuliaSmoothOptimizers/SolverTools.jl).

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