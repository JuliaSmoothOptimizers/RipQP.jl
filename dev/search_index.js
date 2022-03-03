var documenterSearchIndex = {"docs":
[{"location":"reference/#Reference","page":"Reference","title":"Reference","text":"","category":"section"},{"location":"reference/","page":"Reference","title":"Reference","text":"​","category":"page"},{"location":"reference/#Contents","page":"Reference","title":"Contents","text":"","category":"section"},{"location":"reference/","page":"Reference","title":"Reference","text":"​","category":"page"},{"location":"reference/","page":"Reference","title":"Reference","text":"Pages = [\"reference.md\"]","category":"page"},{"location":"reference/","page":"Reference","title":"Reference","text":"​","category":"page"},{"location":"reference/#Index","page":"Reference","title":"Index","text":"","category":"section"},{"location":"reference/","page":"Reference","title":"Reference","text":"​","category":"page"},{"location":"reference/","page":"Reference","title":"Reference","text":"","category":"page"},{"location":"#Home","page":"Home","title":"RipQP.jl documentation","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"This package provides a solver for minimizing convex quadratic problems, of the form:","category":"page"},{"location":"","page":"Home","title":"Home","text":"    min frac12 x^T Q x + c^T x + c_0  st  ell con le A x le ucon  ell le x le u ","category":"page"},{"location":"","page":"Home","title":"Home","text":"where Q is positive semi-definite and the bounds on x and Ax can be infinite.","category":"page"},{"location":"","page":"Home","title":"Home","text":"RipQP uses Interior Point Methods and incorporates several linear algebra and optimization techniques.  It can run in several floating-point systems, and is able to switch between floating-point systems during the  resolution of a problem.","category":"page"},{"location":"","page":"Home","title":"Home","text":"The user should be able to write its own solver to compute the direction of descent that should be used  at each iterate of RipQP.","category":"page"},{"location":"","page":"Home","title":"Home","text":"RipQP can also solve constrained linear least squares problems:","category":"page"},{"location":"","page":"Home","title":"Home","text":"    min frac12  A x - b ^2  st  c_L le C x le c_U  ell le x le u ","category":"page"},{"location":"tutorial/#Tutorial","page":"Tutorial","title":"Tutorial","text":"","category":"section"},{"location":"tutorial/#Input","page":"Tutorial","title":"Input","text":"","category":"section"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"RipQP uses the package QuadraticModels.jl to model convex quadratic problems.","category":"page"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"Here is a basic example:","category":"page"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"using QuadraticModels, LinearAlgebra, SparseMatricesCOO\nQ = [6. 2. 1.\n    2. 5. 2.\n    1. 2. 4.]\nc = [-8.; -3; -3]\nA = [1. 0. 1.\n    0. 2. 1.]\nb = [0.; 3]\nl = [0.;0;0]\nu = [Inf; Inf; Inf]\nQM = QuadraticModel(c, SparseMatrixCOO(tril(Q)), A=SparseMatrixCOO(A), lcon=b, ucon=b, \n                    lvar=l, uvar=u, c0=0., name=\"QM\")","category":"page"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"Once your QuadraticModel is loaded, you can simply solve it RipQP:","category":"page"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"using RipQP\nstats = ripqp(QM)\nprintln(stats)","category":"page"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"The stats output is a GenericExecutionStats.","category":"page"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"It is also possible to use the package QPSReader.jl in order to read convex quadratic problems in MPS or SIF formats:","category":"page"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"using QPSReader, QuadraticModels\nQM = QuadraticModel(readqps(\"QAFIRO.SIF\"))","category":"page"},{"location":"tutorial/#Logging","page":"Tutorial","title":"Logging","text":"","category":"section"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"RipQP displays some logs at each iterate.","category":"page"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"You can deactivate logging with","category":"page"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"stats = ripqp(QM, display = false)","category":"page"},{"location":"tutorial/#Change-configuration-and-tolerances","page":"Tutorial","title":"Change configuration and tolerances","text":"","category":"section"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"The RipQP.InputConfig type allows the user to change the configuration of RipQP. For example, you can use the multi-precision mode without scaling with:","category":"page"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"stats = ripqp(QM, iconf = InputConfig(mode = :multi, scaling = false))","category":"page"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"You can also change the RipQP.InputTol type to change the tolerances for the stopping criteria:","category":"page"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"stats = ripqp(QM, itol = InputTol(max_iter = 100, ϵ_rb = 1.0e-4),\n              iconf = InputConfig(mode = :multi, scaling = false))","category":"page"},{"location":"tutorial/#Save-the-Interior-Point-system","page":"Tutorial","title":"Save the Interior-Point system","text":"","category":"section"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"At every iteration, RipQP solves two linear systems with the default Predictor-Corrector method (the affine system and the  corrector-centering system), or one linear system with the Infeasible Path-Following method.","category":"page"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"To save these systems, you can use:","category":"page"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"w = SystemWrite(write = true, name=\"test_\", kfirst = 4, kgap=3) \nstats1 = ripqp(QM, iconf = InputConfig(w = w))","category":"page"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"This will save one matrix and the associated two right hand sides of the PC method every three iterations starting at  iteration four. Then, you can read the saved files with:","category":"page"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"using DelimitedFiles, MatrixMarket\nK = MatrixMarket.mmread(\"test_K_iter4.mtx\")\nrhs_aff = readdlm(\"test_rhs_iter4_aff.rhs\", Float64)[:]\nrhs_cc =  readdlm(\"test_rhs_iter4_cc.rhs\", Float64)[:] ","category":"page"},{"location":"tutorial/#Advanced:-write-your-own-solver","page":"Tutorial","title":"Advanced: write your own solver","text":"","category":"section"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"You can use your own solver to compute the direction of descent inside RipQP at each iteration. Here is a basic example using the package LDLFactorizations.jl.","category":"page"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"First, you will need a RipQP.SolverParams to define parameters for your solver:","category":"page"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"using RipQP, LinearAlgebra, LDLFactorizations, SparseArrays\n\nstruct K2basicLDLParams{T<:Real} <: SolverParams\n    uplo   :: Symbol # mandatory, tells RipQP which triangle of the augmented system to store\n    ρ      :: T # dual regularization\n    δ      :: T # primal regularization\nend","category":"page"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"Then, you will have to create a type that allocates space for your solver, and a constructor using the following parameters:","category":"page"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"mutable struct PreallocatedDataK2basic{T<:Real, S} <: RipQP.PreallocatedDataAugmented{T, S}\n    D                :: S # temporary top-left diagonal of the K2 system\n    ρ                :: T # dual regularization\n    δ                :: T # primal regularization\n    K                :: SparseMatrixCSC{T,Int} # K2 matrix\n    K_fact           :: LDLFactorizations.LDLFactorization{T,Int,Int,Int} # factorized K2\nend","category":"page"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"Now you need to write a RipQP.PreallocatedData function that returns your type:","category":"page"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"function RipQP.PreallocatedData(sp :: SolverParams, fd :: RipQP.QM_FloatData{T},\n                                id :: RipQP.QM_IntData, itd :: RipQP.IterData{T},\n                                pt :: RipQP.Point{T},\n                                iconf :: InputConfig{Tconf}) where {T<:Real, Tconf<:Real}\n\n    ρ, δ = T(sp.ρ), T(sp.δ)\n    K = spzeros(T, id.ncon+id.nvar, id.ncon + id.nvar)\n    K[1:id.nvar, 1:id.nvar] = .-fd.Q .- ρ .* Diagonal(ones(T, id.nvar))\n    # A = Aᵀ of the input QuadraticModel since we use the upper triangle:\n    K[1:id.nvar, id.nvar+1:end] = fd.A \n    K[diagind(K)[id.nvar+1:end]] .= δ\n\n    K_fact = ldl_analyze(Symmetric(K, :U))\n    @assert sp.uplo == :U # LDLFactorizations does not work with the lower triangle\n    K_fact = ldl_factorize!(Symmetric(K, :U), K_fact)\n    K_fact.__factorized = true\n\n    return PreallocatedDataK2basic(zeros(T, id.nvar),\n                                    ρ,\n                                    δ,\n                                    K, #K\n                                    K_fact #K_fact\n                                    )\nend","category":"page"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"Then, you need to write a RipQP.update_pad! function that will update the RipQP.PreallocatedData struct before computing the direction of descent.","category":"page"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"function RipQP.update_pad!(pad :: PreallocatedDataK2basic{T}, dda :: RipQP.DescentDirectionAllocs{T},\n                           pt :: RipQP.Point{T}, itd :: RipQP.IterData{T},\n                           fd :: RipQP.Abstract_QM_FloatData{T}, id :: RipQP.QM_IntData,\n                           res :: RipQP.Residuals{T}, cnts :: RipQP.Counters,\n                           T0 :: RipQP.DataType) where {T<:Real}\n\n    # update the diagonal of K2\n    pad.D .= -pad.ρ\n    pad.D[id.ilow] .-= pt.s_l ./ itd.x_m_lvar\n    pad.D[id.iupp] .-= pt.s_u ./ itd.uvar_m_x\n    pad.D .-= fd.Q[diagind(fd.Q)]\n    pad.K[diagind(pad.K)[1:id.nvar]] = pad.D\n    pad.K[diagind(pad.K)[id.nvar+1:end]] .= pad.δ\n\n    # factorize K2\n    ldl_factorize!(Symmetric(pad.K, :U), pad.K_fact)\n\nend","category":"page"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"Finally, you need to write a RipQP.solver! function that compute directions of descent. Note that this function solves in-place the linear system by overwriting the direction of descent. That is why the direction of descent dd  countains the right hand side of the linear system to solve.","category":"page"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"function RipQP.solver!(dd :: AbstractVector{T}, pad :: PreallocatedDataK2basic{T},\n                       dda :: RipQP.DescentDirectionAllocsPC{T}, pt :: RipQP.Point{T},\n                       itd :: RipQP.IterData{T}, fd :: RipQP.Abstract_QM_FloatData{T},\n                       id :: RipQP.QM_IntData, res :: RipQP.Residuals{T},\n                       cnts :: RipQP.Counters, T0 :: DataType,\n                       step :: Symbol) where {T<:Real}\n\n    ldiv!(pad.K_fact, dd)\n    return 0\nend","category":"page"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"Then, you can use your solver:","category":"page"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"using QuadraticModels, QPSReader\nqm = QuadraticModel(readqps(\"QAFIRO.SIF\"))\nstats1 = ripqp(qm, iconf = RipQP.InputConfig(sp = K2basicLDLParams(:U, 1.0e-6, 1.0e-6)))","category":"page"},{"location":"API/#RipQP","page":"API","title":"RipQP","text":"","category":"section"},{"location":"API/","page":"API","title":"API","text":"ripqp","category":"page"},{"location":"API/#RipQP.ripqp","page":"API","title":"RipQP.ripqp","text":"stats = ripqp(QM :: QuadraticModel{T0}; iconf :: InputConfig{Int} = InputConfig(),\n              itol :: InputTol{T0, Int} = InputTol(T0),\n              display :: Bool = true) where {T0<:Real}\n\nMinimize a convex quadratic problem. Algorithm stops when the criteria in pdd, rb, and rc are valid. Returns a GenericExecutionStats  containing information about the solved problem.\n\nQM :: QuadraticModel: problem to solve\niconf :: InputConfig{Int}: input RipQP configuration. See RipQP.InputConfig.\nitol :: InputTol{T, Int} input Tolerances for the stopping criteria. See RipQP.InputTol.\ndisplay::Bool: activate/deactivate iteration data display\n\nYou can also use ripqp to solve a LLSModel:\n\nstats = ripqp(LLS :: LLSModel{T0}; iconf :: InputConfig{Int} = InputConfig(),\n              itol :: InputTol{T0, Int} = InputTol(T0),\n              display :: Bool = true) where {T0<:Real}\n\n\n\n\n\n","category":"function"},{"location":"API/#Input-Types","page":"API","title":"Input Types","text":"","category":"section"},{"location":"API/","page":"API","title":"API","text":"InputConfig\nInputTol\nSystemWrite","category":"page"},{"location":"API/#RipQP.InputConfig","page":"API","title":"RipQP.InputConfig","text":"Type to specify the configuration used by RipQP.\n\nmode :: Symbol: should be :mono to use the mono-precision mode, or :multi to use   the multi-precision mode (start in single precision and gradually transitions   to T0)\nscaling :: Bool: activate/deactivate scaling of A and Q in QM0\npresolve :: Bool : activate/deactivate presolve\nnormalize_rtol :: Bool = true : if true, the primal and dual tolerance for the stopping criteria    are normalized by the initial primal and dual residuals\nkc :: Int: number of centrality corrections (set to -1 for automatic computation)\nrefinement :: Symbol : should be :zoom to use the zoom procedure, :multizoom to use the zoom procedure    with multi-precision (then mode should be :multi), ref to use the QP refinement procedure, multiref    to use the QP refinement procedure with multi_precision (then mode should be :multi), or none to avoid    refinements\nsp :: SolverParams : choose a solver to solve linear systems that occurs at each iteration and during the    initialization, see RipQP.SolverParams\nsolve_method :: Symbol : method used to solve the system at each iteration, use solve_method = :PC to    use the Predictor-Corrector algorithm (default), and use solve_method = :IPF to use the Infeasible Path    Following algorithm\nhistory :: Bool : set to true to return the primal and dual norm histories, the primal-dual relative difference   history, and the number of products if using a Krylov method in the solver_specific field of the    GenericExecutionStats\nw :: SystemWrite: configure writing of the systems to solve (no writing is done by default), see RipQP.SystemWrite\n\nThe constructor\n\niconf = InputConfig(; mode :: Symbol = :mono, scaling :: Bool = true, \n                    normalize_rtol :: Bool = true, kc :: I = 0, \n                    refinement :: Symbol = :none, max_ref :: I = 0, \n                    sp :: SolverParams = K2LDLParams(),\n                    solve_method :: Symbol = :PC,\n                    history :: Bool = false, \n                    w :: SystemWrite = SystemWrite()) where {I<:Integer}\n\nreturns a InputConfig struct that shall be used to solve the input QuadraticModel with RipQP.\n\n\n\n\n\n","category":"type"},{"location":"API/#RipQP.InputTol","page":"API","title":"RipQP.InputTol","text":"Type to specify the tolerances used by RipQP.\n\nmax_iter :: Int: maximum number of iterations\nϵ_pdd: relative primal-dual difference tolerance\nϵ_rb: primal tolerance\nϵ_rc: dual tolerance\nmax_iter32, ϵ_pdd32, ϵ_rb32, ϵ_rc32: same as max_iter, ϵ_pdd, ϵ_rb and   ϵ_rc, but used for switching from single precision to double precision. They are   only usefull when mode=:multi\nmax_iter64, ϵ_pdd64, ϵ_rb64, ϵ_rc64: same as max_iter, ϵ_pdd, ϵ_rb and   ϵ_rc, but used for switching from double precision to quadruple precision. They   are only usefull when mode=:multi and T0=Float128\nϵ_rbz : primal transition tolerance for the zoom procedure, (used only if refinement=:zoom)\nϵ_Δx: step tolerance for the current point estimate (note: this criterion   is currently disabled)\nϵ_μ: duality measure tolerance (note: this criterion is currently disabled)\nmax_time: maximum time to solve the QP\n\nThe constructor\n\nitol = InputTol(::Type{T};\n                max_iter :: I = 200, max_iter32 :: I = 40, max_iter64 :: I = 180, \n                ϵ_pdd :: T = 1e-8, ϵ_pdd32 :: T = 1e-2, ϵ_pdd64 :: T = 1e-4, \n                ϵ_rb :: T = 1e-6, ϵ_rb32 :: T = 1e-4, ϵ_rb64 :: T = 1e-5, ϵ_rbz :: T = 1e-3,\n                ϵ_rc :: T = 1e-6, ϵ_rc32 :: T = 1e-4, ϵ_rc64 :: T = 1e-5,\n                ϵ_Δx :: T = 1e-16, ϵ_μ :: T = 1e-9) where {T<:Real, I<:Integer}\n\nInputTol(; kwargs...) = InputTol(Float64; kwargs...)\n\nreturns a InputTol struct that initializes the stopping criteria for RipQP.  The 32 and 64 characters refer to the stopping criteria in :multi mode for the transitions from Float32 to Float64  and Float64 to Float128 (if the input QuadraticModel is in Float128) respectively.\n\n\n\n\n\n","category":"type"},{"location":"API/#RipQP.SystemWrite","page":"API","title":"RipQP.SystemWrite","text":"Type to write the matrix (.mtx format) and the right hand side (.rhs format) of the system to solve at each iteration.\n\nwrite::Bool: activate/deactivate writing of the system \nname::String: name of the sytem to solve \nkfirst::Int: first iteration where a system should be written\nkgap::Int: iteration gap between two problem writings\n\nThe constructor\n\nSystemWrite(; write::Bool = false, name::String = \"\", \n            kfirst::Int = 0, kgap::Int = 1)\n\nreturns a SystemWrite structure that should be used to tell RipQP to save the system.  See the tutorial for more information. \n\n\n\n\n\n","category":"type"},{"location":"API/#Solvers","page":"API","title":"Solvers","text":"","category":"section"},{"location":"API/","page":"API","title":"API","text":"SolverParams\nK2LDLParams\nK2_5LDLParams\nK2KrylovParams\nK2_5KrylovParams\nK2StructuredParams\nK2_5StructuredParams\nK3KrylovParams\nK3SKrylovParams\nK3SStructuredParams\nK3_5KrylovParams\nK3_5StructuredParams\nK1KrylovParams\nK1CholDenseParams\nK2LDLDenseParams\nK1_1StructuredParams\nK1_2StructuredParams","category":"page"},{"location":"API/#RipQP.SolverParams","page":"API","title":"RipQP.SolverParams","text":"Abstract type for tuning the parameters of the different solvers.  Each solver has its own SolverParams type.\n\nThe SolverParams currently implemented within RipQP are:\n\nRipQP.K2LDLParams\nRipQP.K2_5LDLParams\n\n\n\n\n\n","category":"type"},{"location":"API/#RipQP.K2LDLParams","page":"API","title":"RipQP.K2LDLParams","text":"Type to use the K2 formulation with a LDLᵀ factorization, using the package  LDLFactorizations.jl.  The outer constructor \n\nsp = K2LDLParams(; regul = :classic, ρ0 = sqrt(eps()) * 1e5, δ0 = sqrt(eps()) * 1e5)\n\ncreates a RipQP.SolverParams that should be used to create a RipQP.InputConfig. regul = :dynamic uses a dynamic regularization (the regularization is only added if the LDLᵀ factorization  encounters a pivot that has a small magnitude). regul = :none uses no regularization (not recommended). When regul = :classic, the parameters ρ0 and δ0 are used to choose the initial regularization values.\n\n\n\n\n\n","category":"type"},{"location":"API/#RipQP.K2_5LDLParams","page":"API","title":"RipQP.K2_5LDLParams","text":"Type to use the K2.5 formulation with a LDLᵀ factorization, using the package  LDLFactorizations.jl.  The outer constructor \n\nsp = K2_5LDLParams(; regul = :classic, ρ0 = sqrt(eps()) * 1e5, δ0 = sqrt(eps()) * 1e5)\n\ncreates a RipQP.SolverParams that should be used to create a RipQP.InputConfig. regul = :dynamic uses a dynamic regularization (the regularization is only added if the LDLᵀ factorization  encounters a pivot that has a small magnitude). regul = :none uses no regularization (not recommended). When regul = :classic, the parameters ρ0 and δ0 are used to choose the initial regularization values.\n\n\n\n\n\n","category":"type"},{"location":"API/#RipQP.K2KrylovParams","page":"API","title":"RipQP.K2KrylovParams","text":"Type to use the K2 formulation with a Krylov method, using the package  Krylov.jl.  The outer constructor \n\nK2KrylovParams(; uplo = :L, kmethod = :minres, preconditioner = :Identity,\n               rhs_scale = true,\n               atol0 = 1.0e-4, rtol0 = 1.0e-4, \n               atol_min = 1.0e-10, rtol_min = 1.0e-10,\n               ρ0 = sqrt(eps()) * 1e5, δ0 = sqrt(eps()) * 1e5, \n               ρ_min = 1e2 * sqrt(eps()), δ_min = 1e2 * sqrt(eps()),\n               memory = 20)\n\ncreates a RipQP.SolverParams that should be used to create a RipQP.InputConfig. The available methods are:\n\n:minres\n:minres_qlp\n:symmlq\n\nThe list of available preconditioners for this solver is displayed here: RipQP.PreconditionerDataK2.\n\n\n\n\n\n","category":"type"},{"location":"API/#RipQP.K2_5KrylovParams","page":"API","title":"RipQP.K2_5KrylovParams","text":"Type to use the K2.5 formulation with a Krylov method, using the package  Krylov.jl.  The outer constructor \n\nK2_5KrylovParams(; uplo = :L, kmethod = :minres, preconditioner = :Identity,\n                 rhs_scale = true,\n                 atol0 = 1.0e-4, rtol0 = 1.0e-4, \n                 atol_min = 1.0e-10, rtol_min = 1.0e-10,\n                 ρ0 = sqrt(eps()) * 1e5, δ0 = sqrt(eps()) * 1e5, \n                 ρ_min = 1e2 * sqrt(eps()), δ_min = 1e2 * sqrt(eps()),\n                 mem = 20)\n\ncreates a RipQP.SolverParams that should be used to create a RipQP.InputConfig. The available methods are:\n\n:minres\n:minres_qlp\n:symmlq\n\nThe list of available preconditioners for this solver is displayed here: RipQP.PreconditionerDataK2\n\n\n\n\n\n","category":"type"},{"location":"API/#RipQP.K2StructuredParams","page":"API","title":"RipQP.K2StructuredParams","text":"Type to use the K2 formulation with a structured Krylov method, using the package  Krylov.jl. This only works for solving Linear Problems. The outer constructor \n\nK2StructuredParams(; uplo = :L, kmethod = :trimr, rhs_scale = true, \n                   atol0 = 1.0e-4, rtol0 = 1.0e-4,\n                   atol_min = 1.0e-10, rtol_min = 1.0e-10, \n                   ρ_min = 1e2 * sqrt(eps()), δ_min = 1e2 * sqrt(eps()),\n                   mem = 20)\n\ncreates a RipQP.SolverParams that should be used to create a RipQP.InputConfig. The available methods are:\n\n:tricg\n:trimr\n:gpmr\n\nThe list of available preconditioners for this solver is displayed here: RipQP.PreconditionerDataK2. The mem argument sould be used only with gpmr.\n\n\n\n\n\n","category":"type"},{"location":"API/#RipQP.K2_5StructuredParams","page":"API","title":"RipQP.K2_5StructuredParams","text":"Type to use the K2.5 formulation with a structured Krylov method, using the package  Krylov.jl. This only works for solving Linear Problems. The outer constructor \n\nK2_5StructuredParams(; uplo = :L, kmethod = :trimr, rhs_scale = true,\n                     atol0 = 1.0e-4, rtol0 = 1.0e-4,\n                     atol_min = 1.0e-10, rtol_min = 1.0e-10,\n                     ρ0 = sqrt(eps()) * 1e5, δ0 = sqrt(eps()) * 1e5,\n                     ρ_min = 1e2 * sqrt(eps()), δ_min = 1e2 * sqrt(eps()))\n\ncreates a RipQP.SolverParams that should be used to create a RipQP.InputConfig. The available methods are:\n\n:tricg\n:trimr\n:gpmr\n\nThe list of available preconditioners for this solver is displayed here: RipQP.PreconditionerDataK2. The mem argument sould be used only with gpmr.\n\n\n\n\n\n","category":"type"},{"location":"API/#RipQP.K3KrylovParams","page":"API","title":"RipQP.K3KrylovParams","text":"Type to use the K3 formulation with a Krylov method, using the package  Krylov.jl.  The outer constructor \n\nK3KrylovParams(; uplo = :L, kmethod = :qmr, preconditioner = :Identity,\n               rhs_scale = true,\n               atol0 = 1.0e-4, rtol0 = 1.0e-4,\n               atol_min = 1.0e-10, rtol_min = 1.0e-10,\n               ρ0 = sqrt(eps()) * 1e5, δ0 = sqrt(eps()) * 1e5,\n               ρ_min = 1e3 * sqrt(eps()), δ_min = 1e4 * sqrt(eps()),\n               mem = 20)\n\ncreates a RipQP.SolverParams that should be used to create a RipQP.InputConfig. The available methods are:\n\n:qmr\n:bicgstab\n:usymqr\n\n\n\n\n\n","category":"type"},{"location":"API/#RipQP.K3SKrylovParams","page":"API","title":"RipQP.K3SKrylovParams","text":"Type to use the K3S formulation with a Krylov method, using the package  Krylov.jl.  The outer constructor \n\nK3SKrylovParams(; uplo = :L, kmethod = :minres, preconditioner = :Identity,\n                 rhs_scale = true,\n                 atol0 = 1.0e-4, rtol0 = 1.0e-4,\n                 atol_min = 1.0e-10, rtol_min = 1.0e-10,\n                 ρ0 = sqrt(eps()) * 1e5, δ0 = sqrt(eps()) * 1e5,\n                 ρ_min = 1e3 * sqrt(eps()), δ_min = 1e4 * sqrt(eps()),\n                 mem = 20)\n\ncreates a RipQP.SolverParams that should be used to create a RipQP.InputConfig. The available methods are:\n\n:minres\n:minres_qlp\n:symmlq\n\n\n\n\n\n","category":"type"},{"location":"API/#RipQP.K3SStructuredParams","page":"API","title":"RipQP.K3SStructuredParams","text":"Type to use the K3S formulation with a Krylov method, using the package  Krylov.jl.  The outer constructor \n\nK3SStructuredParams(; uplo = :U, kmethod = :trimr, rhs_scale = true,\n                     atol0 = 1.0e-4, rtol0 = 1.0e-4, \n                     atol_min = 1.0e-10, rtol_min = 1.0e-10,\n                     ρ0 =  sqrt(eps()) * 1e3, δ0 = sqrt(eps()) * 1e4,\n                     ρ_min = 1e4 * sqrt(eps()), δ_min = 1e4 * sqrt(eps()),\n                     mem = 20)\n\ncreates a RipQP.SolverParams that should be used to create a RipQP.InputConfig. The available methods are:\n\n:tricg\n:trimr\n:gpmr\n\nThe mem argument sould be used only with gpmr.\n\n\n\n\n\n","category":"type"},{"location":"API/#RipQP.K3_5KrylovParams","page":"API","title":"RipQP.K3_5KrylovParams","text":"Type to use the K3.5 formulation with a Krylov method, using the package  Krylov.jl.  The outer constructor \n\nK3_5KrylovParams(; uplo = :L, kmethod = :minres, preconditioner = :Identity,\n                 rhs_scale = true,\n                 atol0 = 1.0e-4, rtol0 = 1.0e-4,\n                 atol_min = 1.0e-10, rtol_min = 1.0e-10,\n                 ρ0 = sqrt(eps()) * 1e5, δ0 = sqrt(eps()) * 1e5,\n                 ρ_min = 1e3 * sqrt(eps()), δ_min = 1e4 * sqrt(eps()),\n                 mem = 20)\n\ncreates a RipQP.SolverParams that should be used to create a RipQP.InputConfig. The available methods are:\n\n:minres\n:minres_qlp\n:symmlq\n\n\n\n\n\n","category":"type"},{"location":"API/#RipQP.K3_5StructuredParams","page":"API","title":"RipQP.K3_5StructuredParams","text":"Type to use the K3.5 formulation with a Krylov method, using the package  Krylov.jl.  The outer constructor \n\nK3_5StructuredParams(; uplo = :U, kmethod = :trimr, rhs_scale = true,\n                     atol0 = 1.0e-4, rtol0 = 1.0e-4, \n                     atol_min = 1.0e-10, rtol_min = 1.0e-10,\n                     ρ0 =  sqrt(eps()) * 1e3, δ0 = sqrt(eps()) * 1e4,\n                     ρ_min = 1e4 * sqrt(eps()), δ_min = 1e4 * sqrt(eps()),\n                     mem = 20)\n\ncreates a RipQP.SolverParams that should be used to create a RipQP.InputConfig. The available methods are:\n\n:tricg\n:trimr\n:gpmr\n\nThe mem argument sould be used only with gpmr.\n\n\n\n\n\n","category":"type"},{"location":"API/#RipQP.K1KrylovParams","page":"API","title":"RipQP.K1KrylovParams","text":"Type to use the K1 formulation with a Krylov method, using the package  Krylov.jl.  The outer constructor \n\nK1KrylovParams(; uplo = :L, kmethod = :cg, preconditioner = :Identity,\n               rhs_scale = true,\n               atol0 = 1.0e-4, rtol0 = 1.0e-4, \n               atol_min = 1.0e-10, rtol_min = 1.0e-10,\n               ρ0 = sqrt(eps()) * 1e5, δ0 = sqrt(eps()) * 1e5, \n               ρ_min = 1e2 * sqrt(eps()), δ_min = 1e2 * sqrt(eps()),\n               mem = 20)\n\ncreates a RipQP.SolverParams that should be used to create a RipQP.InputConfig. The available methods are:\n\n:cg\n:cg_lanczos\n:cr\n:minres\n:minres_qlp\n:symmlq\n\n\n\n\n\n","category":"type"},{"location":"API/#RipQP.K1CholDenseParams","page":"API","title":"RipQP.K1CholDenseParams","text":"Type to use the K1 formulation with a dense Cholesky factorization. The input QuadraticModel should have lcon .== ucon. The outer constructor \n\nsp = K1CholDenseParams(; ρ0 = sqrt(eps()) * 1e5, δ0 = sqrt(eps()) * 1e5)\n\ncreates a RipQP.SolverParams that should be used to create a RipQP.InputConfig.\n\n\n\n\n\n","category":"type"},{"location":"API/#RipQP.K2LDLDenseParams","page":"API","title":"RipQP.K2LDLDenseParams","text":"Type to use the K2 formulation with a LDLᵀ factorization.  The outer constructor \n\nsp = K2LDLDenseParams(; ρ0 = sqrt(eps()) * 1e5, δ0 = sqrt(eps()) * 1e5)\n\ncreates a RipQP.SolverParams that should be used to create a RipQP.InputConfig.\n\n\n\n\n\n","category":"type"},{"location":"API/#RipQP.K1_1StructuredParams","page":"API","title":"RipQP.K1_1StructuredParams","text":"Type to use the K1.1 formulation with a structured Krylov method, using the package  Krylov.jl. This only works for solving Linear Problems. The outer constructor \n\nK1_1StructuredParams(; uplo = :L, kmethod = :lsqr, rhs_scale = true,\n                     atol0 = 1.0e-4, rtol0 = 1.0e-4,\n                     atol_min = 1.0e-10, rtol_min = 1.0e-10, \n                     ρ_min = 1e3 * sqrt(eps()), δ_min = 1e4 * sqrt(eps()),\n                     mem = 20)\n\ncreates a RipQP.SolverParams that should be used to create a RipQP.InputConfig. The available methods are:\n\n:lslq\n:lsqr\n:lsmr\n\n\n\n\n\n","category":"type"},{"location":"API/#RipQP.K1_2StructuredParams","page":"API","title":"RipQP.K1_2StructuredParams","text":"Type to use the K1.2 formulation with a structured Krylov method, using the package  Krylov.jl. This only works for solving Linear Problems. The outer constructor \n\nK1_2StructuredParams(; uplo = :L, kmethod = :craig, rhs_scale = true,\n                     atol0 = 1.0e-4, rtol0 = 1.0e-4,\n                     atol_min = 1.0e-10, rtol_min = 1.0e-10, \n                     ρ_min = 1e3 * sqrt(eps()), δ_min = 1e4 * sqrt(eps()),\n                     mem = 20)\n\ncreates a RipQP.SolverParams that should be used to create a RipQP.InputConfig. The available methods are:\n\n:lnlq\n:craig\n:craigmr\n\n\n\n\n\n","category":"type"},{"location":"API/#Preconditioners","page":"API","title":"Preconditioners","text":"","category":"section"},{"location":"API/","page":"API","title":"API","text":"PreconditionerDataK2","category":"page"},{"location":"API/#RipQP.PreconditionerDataK2","page":"API","title":"RipQP.PreconditionerDataK2","text":"Abstract type that defines preconditioners for the K2 formulation. The available preconditioners are:\n\nIdentity\nJacobi \n\n\n\n\n\n","category":"type"}]
}
