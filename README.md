# RipQP

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4309783.svg)](https://doi.org/10.5281/zenodo.4309783)
![CI](https://github.com/JuliaSmoothOptimizers/RipQP.jl/workflows/CI/badge.svg?branch=main)
[![Cirrus CI - Base Branch Build Status](https://img.shields.io/cirrus/github/JuliaSmoothOptimizers/RipQP.jl?logo=Cirrus%20CI)](https://cirrus-ci.com/github/JuliaSmoothOptimizers/RipQP.jl)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://JuliaSmoothOptimizers.github.io/RipQP.jl/dev)
[![](https://img.shields.io/badge/docs-stable-3f51b5.svg)](https://JuliaSmoothOptimizers.github.io/RipQP.jl/stable)
[![codecov](https://codecov.io/gh/JuliaSmoothOptimizers/RipQP.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/JuliaSmoothOptimizers/RipQP.jl)

A package to optimize linear and quadratic problems in QuadraticModel format
(see https://github.com/JuliaSmoothOptimizers/QuadraticModels.jl).

By default, RipQP iterates in the floating-point type of its input QuadraticModel, but it can also perform operations in several floating-point systems if some parameters are modified (see the [documentation](https://JuliaSmoothOptimizers.github.io/RipQP.jl/stable) for more information).

# Basic usage

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
stats = ripqp(qm, mode=:multi, itol = InputTol(max_iter=100))
```

## Bug reports and discussions

If you think you found a bug, feel free to open an [issue](https://github.com/JuliaSmoothOptimizers/RipQP.jl/issues).
Focused suggestions and requests can also be opened as issues. Before opening a pull request, start an issue or a discussion on the topic, please.

If you want to ask a question not suited for a bug report, feel free to start a discussion [here](https://github.com/JuliaSmoothOptimizers/Organization/discussions). This forum is for general discussion about this repository and the [JuliaSmoothOptimizers](https://github.com/JuliaSmoothOptimizers) organization, so questions about any of our packages are welcome.
