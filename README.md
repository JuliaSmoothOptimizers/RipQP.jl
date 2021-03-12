# RipQP

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4309783.svg)](https://doi.org/10.5281/zenodo.4309783)
![CI](https://github.com/JuliaSmoothOptimizers/RipQP.jl/workflows/CI/badge.svg?branch=master)
[![Cirrus CI - Base Branch Build Status](https://img.shields.io/cirrus/github/JuliaSmoothOptimizers/RipQP.jl?logo=Cirrus%20CI)](https://cirrus-ci.com/github/JuliaSmoothOptimizers/RipQP.jl)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://JuliaSmoothOptimizers.github.io/RipQP.jl/dev)
[![](https://img.shields.io/badge/docs-stable-3f51b5.svg)](https://JuliaSmoothOptimizers.github.io/RipQP.jl/stable)
[![codecov](https://codecov.io/gh/JuliaSmoothOptimizers/RipQP.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/JuliaSmoothOptimizers/RipQP.jl)

A package to optimize linear and quadratic problems in QuadraticModel format
(see https://github.com/JuliaSmoothOptimizers/QuadraticModels.jl).

The ripQP function can work in mono mode (double precision only), or in multi
mode (single precision, then double precision).
Each iteration in single precision counts for 1 iteration, and each iteration in
double precision counts for 4 iterations.

# Usage

In this example, we use QPSReader to read a quadratic problem (QAFIRO) from the
Maros and Meszaros dataset.

```julia
using QPSReader, QuadraticModels
using RipQP
qps = readqps("QAFIRO.SIF")
qm = QuadraticModel(qps)
stats = ripqp(qm)
```

To use the multi precision mode (default to :mono) and change the maximum number of iterations:
```julia
stats = ripqp(qm, iconf = input_config(mode=:multi), itol = input_tol(max_iter=100))
```
