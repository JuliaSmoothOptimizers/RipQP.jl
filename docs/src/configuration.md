# Configuration

This section gives more information on how to choose a configuration with [`RipQP.InputConfig`](@ref).

## Solver

The `solver` parameter allows the user to choose a solver that will solve the linear systems during the initialization and at each iteration.

- `solver = :K2` solves the K2 system using a LDLᵀ factorization
- `solver = :K2_5` solves the K2.5 system using a LDLᵀ factorization

## Solve Method

The `solve_method` parameter allows the user to choose the Interior Point Algorithm used.

- `solve_method = :PC` uses a Predictor-Corrector Algorithm that solves two linear systems per iteration
- `solve_method = :IPF` uses an Infeasible Path Following Algorithm that solves one linear system per iteration

