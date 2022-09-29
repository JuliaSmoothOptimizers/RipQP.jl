# [RipQP.jl documentation](@id Home)

This package provides a solver for minimizing convex quadratic problems, of the form:

```math
    \min \frac{1}{2} x^T Q x + c^T x + c_0, ~~~~ s.t. ~~ \ell con \le A x \le ucon, ~~ \ell \le x \le u, 
```

where Q is positive semi-definite and the bounds on x and Ax can be infinite.

RipQP uses Interior Point Methods and incorporates several linear algebra and optimization techniques. 
It can run in several floating-point systems, and is able to switch between floating-point systems during the 
resolution of a problem.

The user should be able to write its own solver to compute the direction of descent that should be used 
at each iterate of RipQP.

RipQP can also solve constrained linear least squares problems:

```math
    \min \frac{1}{2} \| A x - b \|^2, ~~~~ s.t. ~~ c_L \le C x \le c_U, ~~ \ell \le x \le u. 
```

## Bug reports and discussions

If you think you found a bug, feel free to open an [issue](https://github.com/JuliaSmoothOptimizers/RipQP.jl/issues).
Focused suggestions and requests can also be opened as issues. Before opening a pull request, start an issue or a discussion on the topic, please.

If you want to ask a question not suited for a bug report, feel free to start a discussion [here](https://github.com/JuliaSmoothOptimizers/Organization/discussions). This forum is for general discussion about this repository and the [JuliaSmoothOptimizers](https://github.com/JuliaSmoothOptimizers) organization, so questions about any of our packages are welcome.
