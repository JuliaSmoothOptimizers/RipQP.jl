# [RipQP.jl documentation](@id Home)

This package provides a solver for minimizing convex quadratic problems, of the form:

```math
    \min \frac{1}{2} x^T Q x + c^T x + c_0, ~~~~ s.t. ~~ lcon \le A x \le ucon, ~~ l \le x \le u, 
```

where Q is positive semi-definite and the bounds on x and Ax can be infinite.

RipQP uses Interior Point Methods and incorporates several linear algebra and optimization techniques. 
It can run in several floating-point systems, and is able to switch between floating-point systems during the 
resolution of a problem.

The user should be able to write its own solver to compute the direction of descent that should be used 
at each iterate of RipQP.

Note: RipQP does not handle problems with fixed variables yet.