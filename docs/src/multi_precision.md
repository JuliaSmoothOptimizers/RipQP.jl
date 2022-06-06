# Multi-precision

## Solving precision

You can use RipQP in several floating-point systems.
The algorithm will run in the precision of the input `QuadraticModel`.
For example, if you have a `QuadraticModel` `qm32` in `Float32`, then

```julia
stats = ripqp(qm32)
```

will solve this problem in `Float32`.
The stopping criteria of [`RipQP.InputTol`](@ref) will be adapted to the preicison of the `QuadraticModel` to solve.

## Multi-precision

You can use the multi-precision mode to solve a problem with a warm-start in a lower floating-point system.
[`RipQP.InputTol`](@ref) contains intermediate parameters that are used to decide when to transition from a lower precision to a higher precision.

```julia
stats = ripqp(qm, mode = :multi, itol = InputTol(ϵ_pdd32 = 1.0e-2))
```

## Refinement of the Quadratic Problem

Instead of just increasing the precision of the algorithm for the transition between precisions, it is possible to solve a refined quadratic problem.

References:
* T. Weber, S. Sager, A. Gleixner [*Solving quadratic programs to high precision using scaled iterative refinement*](https://doi.org/10.1007/s12532-019-00154-6), Mathematical Programming Computation, 49(6), pp. 421-455, 2019.
* D. Ma, L. Yang, R. M. T. Fleming, I. Thiele, B. O. Palsson, M. A. Saunders [*Reliable and efficient solution of genome-scale models of Metabolism and macromolecular Expression*](https://doi.org/10.1038/srep40863), Scientific Reports 7, 40863, 2017.

```julia
stats = ripqp(qm, mode = :multiref, itol = InputTol(ϵ_pdd32 = 1.0e-2))
stats = ripqp(qm, mode = :multizoom, itol = InputTol(ϵ_pdd32 = 1.0e-2))
```

The two presented algorithms follow the procedure described in each of the two above references.

## Switching solvers when increasing precision

Instead of using the same solver after the transition between two floating-point systems, it is possible to switch to another solver, if a conversion function from the old solver to the new solvers is implemented.

```julia
stats = ripqp(qm, mode = :multiref, solve_method = IPF(),
              sp = K2LDLParams(),
              sp2 = K2KrylovParams(uplo = :U, preconditioner = LDL(T = Float32)))
# start with K2LDL in Float32, then transition to Float64, 
# then use K2Krylov in Float64 with a LDL preconditioner in Float32.
```