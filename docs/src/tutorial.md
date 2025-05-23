# Tutorial

## Input

RipQP uses the package [QuadraticModels.jl](https://github.com/JuliaSmoothOptimizers/QuadraticModels.jl) to model
convex quadratic problems.

Here is a basic example:

```@example QM
using QuadraticModels, LinearAlgebra, SparseMatricesCOO
Q = [6. 2. 1.
     2. 5. 2.
     1. 2. 4.]
c = [-8.; -3; -3]
A = [1. 0. 1.
     0. 2. 1.]
b = [0.; 3]
l = [-1.0;0;0]
u = [Inf; Inf; Inf]
QM = QuadraticModel(
  c,
  SparseMatrixCOO(tril(Q)),
  A=SparseMatrixCOO(A),
  lcon=b,
  ucon=b,
  lvar=l,
  uvar=u,
  c0=0.,
  name="QM"
)
```

Once your `QuadraticModel` is loaded, you can simply solve it RipQP:

```@example QM
using RipQP
stats = ripqp(QM)
println(stats)
```

The `stats` output is a
[GenericExecutionStats](https://jso.dev/SolverCore.jl/stable/reference/#SolverCore.GenericExecutionStats).

It is also possible to use the package [QPSReader.jl](https://github.com/JuliaSmoothOptimizers/QPSReader.jl) in order to
read convex quadratic problems in MPS or SIF formats: (download [QAFIRO](https://raw.githubusercontent.com/JuliaSmoothOptimizers/RipQP.jl/main/test/QAFIRO.SIF))

```@example QM
using QPSReader, QuadraticModels
QM = QuadraticModel(readqps("assets/QAFIRO.SIF"))
```

## Logging

RipQP displays some logs at each iterate.

```@example QM
stats = ripqp(QM)
println(stats)
```

You can deactivate logging with

```@example QM
stats = ripqp(QM, display = false)
println(stats)
```

It is also possible to get a history of several quantities such as the primal and dual residuals and the relative primal-dual gap. These quantites are available in the dictionary `solver_specific` of the `stats`.

```@example QM
stats = ripqp(QM, display = false, history = true)
pddH = stats.solver_specific[:pddH]
```

## Change configuration and tolerances

You can use `RipQP` without presolve and scaling with:

```julia
stats = ripqp(QM, ps=false, scaling = false)
```

You can also change the [`RipQP.InputTol`](https://jso.dev/RipQP.jl/stable/API/#RipQP.InputTol) type to change the tolerances for the stopping criteria:

```@example QM
stats = ripqp(QM, itol = InputTol(max_iter = 5))
println(stats)
```

## Save the Interior-Point system

At every iteration, RipQP solves two linear systems with the default Predictor-Corrector method (the affine system and the corrector-centering system), or one linear system with the Infeasible Path-Following method.

To save these systems, you can use:

```julia
w = SystemWrite(write = true, name="test_", kfirst = 4, kgap=3)
stats = ripqp(QM, w = w)
```

This will save one matrix and the associated two right hand sides of the PC method every three iterations starting at iteration four.
Then, you can read the saved files with:

```julia
using DelimitedFiles, MatrixMarket
K = MatrixMarket.mmread("test_K_iter4.mtx")
rhs_aff = readdlm("test_rhs_iter4_aff.rhs", Float64)[:]
rhs_cc =  readdlm("test_rhs_iter4_cc.rhs", Float64)[:];
```

## Timers

You can see the elapsed time with:

```julia
stats.elapsed_time
```

For more advance timers you can use [TimerOutputs.jl](https://github.com/KristofferC/TimerOutputs.jl):

```julia
using TimerOutputs
TimerOutputs.enable_debug_timings(RipQP)
reset_timer!(RipQP.to)
stats = ripqp(QM)
TimerOutputs.complement!(RipQP.to) # print complement of timed sections
show(RipQP.to, sortby = :firstexec)
```
