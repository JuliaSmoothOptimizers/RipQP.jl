# RipQP

A package to optimize linear and quadratic problems in QuadraticModels format
(see QuadraticModels.jl).

The ripQP function can work in mono mode (double precision only), or in multi
mode (single precision, then double precision).

# Usage

In this example, we use QPSReader to read a quadratic problem (QAFIRO) from the
Maros and Meszaros dataset.

```julia
using QPSReader, QuadraticModels
using RipQP
qps = readqps("QAFIRO.SIF")
qm = QuadraticModel(qps)
stats = ripQP(qm)
```
