module RipQP

using LinearAlgebra, SparseArrays, Statistics

using LDLFactorizations, NLPModels, QuadraticModels, SolverTools

export ripQP

include("starting_points.jl")
include("scaling.jl")
include("sparse_toolbox.jl")
include("iterations.jl")

function ripQP(QM0; max_iter=800, ϵ_pdd=1e-8, ϵ_rb=1e-6, ϵ_rc=1e-6,
               tol_Δx=1e-16, ϵ_μ=1e-9, max_time=1200., scaling=true,
               display=true)

    start_time = time()
    elapsed_time = 0.0
    QM = SlackModel(QM0)

    # get variables from QuadraticModel
    lvar, uvar = QM.meta.lvar, QM.meta.uvar
    n_cols = length(lvar)
    Oc = zeros(n_cols)
    ilow, iupp = [QM.meta.ilow; QM.meta.irng], [QM.meta.iupp; QM.meta.irng] # finite bounds index
    irng = QM.meta.irng
    ifix = QM.meta.ifix
    c = grad(QM, Oc)
    A = jac(QM, Oc)
    A = dropzeros!(A)
    T = eltype(A)
    Arows, Acols, Avals = findnz(A)
    n_rows, n_cols = size(A)
    @assert QM.meta.lcon == QM.meta.ucon # equality constraint (Ax=b)
    b = QM.meta.lcon
    Q = hess(QM, Oc)  # lower triangular
    Q = dropzeros!(Q)
    Qrows, Qcols, Qvals = findnz(Q)
    c0 = obj(QM, Oc)

    if scaling
        Arows, Acols, Avals, Qrows, Qcols, Qvals,
        c, b, lvar, uvar, d1, d2, d3 = scaling_Ruiz!(Arows, Acols, Avals, Qrows, Qcols, Qvals,
                                                     c, b, lvar, uvar, n_rows, n_cols, T(1.0e-3))
    end

#     cNorm = norm(c)
#     bNorm = norm(b)
#     ANorm = norm(Avals)  # Frobenius norm after scaling; could be computed while scaling?
#     QNorm = norm(Qvals)
    n_low, n_upp = length(ilow), length(iupp) # number of finite constraints

    #change types
    T = Float32
    Qvals32 = Array{T}(Qvals)
    Avals32 = Array{T}(Avals)
    c32 = Array{T}(c)
    c032 = T(c0)
    b32 = Array{T}(b)
    lvar32 = Array{T}(lvar)
    uvar32 = Array{T}(uvar)
    ϵ_pdd32 = T(1e-2)
    ϵ_rb32 = T(1e-2)
    ϵ_rc32 = T(1e-2)
    tol_Δx32 = T(tol_Δx)
    ϵ_μ32 = T(ϵ_μ)

    # init regularization values
    ρ, δ = T(sqrt(eps())*1e5), T(sqrt(eps())*1e5)
    ρ_min, δ_min = T(sqrt(eps(T))*1e0), T(sqrt(eps(T))*1e0)
    c_catch = zero(Int) # to avoid endless loop
    c_pdd = zero(Int) # avoid too small δ_min

    J_augmrows = vcat(Qcols, Acols, n_cols+1:n_cols+n_rows, 1:n_cols)
    J_augmcols = vcat(Qrows, Arows.+n_cols, n_cols+1:n_cols+n_rows, 1:n_cols)
    tmp_diag = -T(1.0e-2) .* ones(T, n_cols)
    J_augmvals = vcat(-Qvals32, Avals32, δ.*ones(T, n_rows), tmp_diag)
    J_augm = sparse(J_augmrows, J_augmcols, J_augmvals)
    diagind_J = get_diag_sparseCSC(J_augm)
    diag_Q = get_diag_sparseCOO(Qrows, Qcols, Qvals32, n_cols)

    k = 0
    Δ_aff = zeros(T, n_cols+n_rows+n_low+n_upp)
    Δ_cc = zeros(T, n_cols+n_rows+n_low+n_upp)
    Δ = zeros(T, n_cols+n_rows+n_low+n_upp)
    Δ_xλ = zeros(T, n_cols+n_rows)

    x, λ, s_l, s_u, J_fact, J_P, Qx, ATλ,
    x_m_lvar, uvar_m_x, Δ_xλ = @views starting_points(Qrows, Qcols, Qvals32, Arows, Acols, Avals32,
                                                      b32, c32, lvar32, uvar32, ilow, iupp, QM.meta.irng,
                                                      J_augm , n_rows, n_cols, Δ_xλ)


    Qx = mul_Qx_COO!(Qx, Qrows, Qcols, Qvals32, x)
    ATλ = mul_ATλ_COO!(ATλ, Arows, Acols, Avals32, λ)
    Ax = zeros(T,  n_rows)
    Ax = mul_Ax_COO!(Ax, Arows, Acols, Avals32, x)
    rb = Ax - b32
    rc = -Qx + ATλ + s_l - s_u - c32

    x_m_lvar .= @views x[ilow] .- lvar32[ilow]
    uvar_m_x .= @views uvar32[iupp] .- x[iupp]
    μ = @views compute_μ(x_m_lvar, uvar_m_x, s_l[ilow], s_u[iupp], n_low, n_upp)

    x_m_l_αΔ_aff = zeros(T, n_low) # x-lvar + αΔ_aff
    u_m_x_αΔ_aff = zeros(T, n_upp) # uvar-x + αΔ_aff
    s_l_αΔ_aff = zeros(T, n_low) # s_l + αΔ_aff
    s_u_αΔ_aff = zeros(T, n_upp) # s_l + αΔ_aff
    rxs_l, rxs_u = zeros(T, n_low), zeros(T, n_upp)

    # stopping criterion
    xTQx_2 = x' * Qx / 2
    cTx = c32' * x
    pri_obj = xTQx_2 + cTx + c032
    dual_obj = b32' * λ - xTQx_2 + view(s_l,ilow)'*view(lvar32,ilow) -
                    view(s_u,iupp)'*view(uvar32,iupp) +c032
    pdd = abs(pri_obj - dual_obj ) / (one(T) + abs(pri_obj))
#     rcNorm, rbNorm = norm(rc), norm(rb)
#     optimal = pdd < ϵ_pdd && rbNorm < ϵ_rb && rcNorm < ϵ_rc
    rcNorm, rbNorm = norm(rc, Inf), norm(rb, Inf)
    tol_rb32, tol_rc32 = ϵ_rb32*(one(T) + rbNorm), ϵ_rc32*(one(T) + rcNorm)
    tol_rb, tol_rc = ϵ_rb*(one(Float64) + Float64(rbNorm)), ϵ_rc*(one(Float64) + Float64(rcNorm))
    optimal = pdd < ϵ_pdd32 && rbNorm < tol_rb32 && rcNorm < tol_rc32

    l_pdd = zeros(T, 6)
    mean_pdd = one(T)

    n_Δx = zero(T)
    small_Δx, small_μ = false, μ < ϵ_μ32
    Δt = time() - start_time
    tired = Δt > max_time

    # display
    if display == true
        @info log_header([:k, :pri_obj, :pdd, :rbNorm, :rcNorm, :n_Δx, :α_pri, :α_du, :μ],
                         [Int, T, T, T, T, T, T, T, T, T],
                         hdr_override=Dict(:k => "iter", :pri_obj => "obj", :pdd => "rgap",
                                           :rbNorm => "‖rb‖", :rcNorm => "‖rc‖",
                                           :n_Δx => "‖Δx‖"))
        @info log_row(Any[k, pri_obj, pdd, rbNorm, rcNorm, n_Δx, zero(T), zero(T), μ])
    end

    # iters Float 32
    x, λ, s_l, s_u, x_m_lvar, uvar_m_x,
        rc, rb, rcNorm, rbNorm, Qx, ATλ,
        Ax, xTQx_2, cTx, pri_obj, dual_obj,
        pdd, l_pdd, mean_pdd, n_Δx, Δt,
        tired, optimal, μ, k, ρ, δ,
        ρ_min, δ_min, J_augm, J_fact,
        c_catch, c_pdd  = iter_mehrotraPC!(x, λ, s_l, s_u, x_m_lvar, uvar_m_x, lvar32, uvar32,
                                          ilow, iupp, n_rows, n_cols,n_low, n_upp,
                                          Arows, Acols, Avals32, Qrows, Qcols, Qvals32, c032,
                                          c32, b32, rc, rb, rcNorm, rbNorm, tol_rb32, tol_rc32,
                                          Qx, ATλ, Ax, xTQx_2, cTx, pri_obj, dual_obj,
                                          pdd, l_pdd, mean_pdd, n_Δx, small_Δx, small_μ,
                                          Δt, tired, optimal, μ, k, ρ, δ, ρ_min, δ_min,
                                          J_augm, J_fact, J_P, diagind_J, diag_Q, tmp_diag,
                                          Δ_aff, Δ_cc, Δ, Δ_xλ, s_l_αΔ_aff, s_u_αΔ_aff,
                                          x_m_l_αΔ_aff, u_m_x_αΔ_aff, rxs_l, rxs_u,
                                          20, ϵ_pdd32, ϵ_μ32, ϵ_rc32, ϵ_rb32, tol_Δx32,
                                          start_time, max_time, c_catch, c_pdd, display)

    # conversions to Float64
    T = Float64
    x, λ, s_l, s_u = convert(Array{T}, x), convert(Array{T}, λ), convert(Array{T}, s_l), convert(Array{T}, s_u)
    x_m_lvar, uvar_m_x = convert(Array{T}, x_m_lvar), convert(Array{T}, uvar_m_x)
    rc, rb = convert(Array{T}, rc), convert(Array{T}, rb)
    rcNorm, rbNorm = convert(T, rcNorm), convert(T, rbNorm)
    Qx, ATλ, Ax = convert(Array{T}, Qx), convert(Array{T}, ATλ), convert(Array{T}, Ax)
    xTQx_2, cTx = convert(T, xTQx_2), convert(T, cTx)
    pri_obj, dual_obj = convert(T, pri_obj), convert(T, dual_obj)
    pdd, l_pdd, mean_pdd = convert(T, pdd), convert(Array{T}, l_pdd), convert(T, mean_pdd)
    n_Δx, μ = convert(T, n_Δx), convert(T, μ)
    ρ, δ = convert(T, ρ), convert(T, δ)
    ρ /= 10
    δ /= 10
    J_augm = convert(SparseMatrixCSC{T,Int64}, J_augm)
    J_P = LDLFactorizations.LDLFactorization(J_P.__analyzed, J_P.__factorized, J_P.__upper,
                              J_P.n, J_P.parent, J_P.Lnz, J_P.flag, J_P.P,
                              J_P.pinv, J_P.Lp, J_P.Cp, J_P.Ci, J_P.Li,
                              Array{T}(J_P.Lx), Array{T}(J_P.d), Array{T}(J_P.Y), J_P.pattern)
    J_fact = LDLFactorizations.LDLFactorization(J_fact.__analyzed, J_fact.__factorized, J_fact.__upper,
                              J_fact.n, J_fact.parent, J_fact.Lnz, J_fact.flag, J_fact.P,
                              J_fact.pinv, J_fact.Lp, J_fact.Cp, J_fact.Ci, J_fact.Li,
                              Array{T}(J_fact.Lx), Array{T}(J_fact.d), Array{T}(J_fact.Y), J_fact.pattern)
    Δ_aff, Δ_cc, Δ = convert(Array{T}, Δ_aff), convert(Array{T}, Δ_cc), convert(Array{T}, Δ)
    Δ_xλ, rxs_l, rxs_u = convert(Array{T}, Δ_xλ), convert(Array{T}, rxs_l), convert(Array{T}, rxs_u)
    s_l_αΔ_aff, s_u_αΔ_aff = convert(Array{T}, s_l_αΔ_aff), convert(Array{T}, s_u_αΔ_aff)
    x_m_l_αΔ_aff, u_m_x_αΔ_aff = convert(Array{T}, x_m_l_αΔ_aff), convert(Array{T}, u_m_x_αΔ_aff)
    diag_Q, tmp_diag = convert(Array{T}, diag_Q), convert(Array{T}, tmp_diag)

    ρ_min, δ_min = T(sqrt(eps())*1e-5), T(sqrt(eps())*1e0)
    optimal = pdd < ϵ_pdd && rbNorm < tol_rb && rcNorm < tol_rc

    # iters Float64
    x, λ, s_l, s_u, x_m_lvar, uvar_m_x,
        rc, rb, rcNorm, rbNorm, Qx, ATλ,
        Ax, xTQx_2, cTx, pri_obj, dual_obj,
        pdd, l_pdd, mean_pdd, n_Δx, Δt,
        tired, optimal, μ, k, ρ, δ,
        ρ_min, δ_min, J_augm, J_fact,
        c_catch, c_pdd  = iter_mehrotraPC!(x, λ, s_l, s_u, x_m_lvar, uvar_m_x, lvar, uvar,
                                          ilow, iupp, n_rows, n_cols,n_low, n_upp,
                                          Arows, Acols, Avals, Qrows, Qcols, Qvals, c0,
                                          c, b, rc, rb, rcNorm, rbNorm, tol_rb, tol_rc,
                                          Qx, ATλ, Ax, xTQx_2, cTx, pri_obj, dual_obj,
                                          pdd, l_pdd, mean_pdd, n_Δx, small_Δx, small_μ,
                                          Δt, tired, optimal, μ, k, ρ, δ, ρ_min, δ_min,
                                          J_augm, J_fact, J_P, diagind_J, diag_Q, tmp_diag,
                                          Δ_aff, Δ_cc, Δ, Δ_xλ, s_l_αΔ_aff, s_u_αΔ_aff,
                                          x_m_l_αΔ_aff, u_m_x_αΔ_aff, rxs_l, rxs_u,
                                          max_iter, ϵ_pdd, ϵ_μ, ϵ_rc, ϵ_rb, tol_Δx,
                                          start_time, max_time, c_catch, c_pdd, display)

    if k>= max_iter
        status = :max_iter
    elseif tired
        status = :max_time
    elseif optimal
        status = :acceptable
    else
        status = :unknown
    end

    if scaling
        d1 = convert(Array{T}, d1)
        d2 = convert(Array{T}, d2)
        d3 = convert(Array{T}, d3)
        x .*= d2 .* d3
        for i=1:length(Qrows)
            Qvals[i] /= d2[Qrows[i]] * d2[Qcols[i]] * d3[Qrows[i]] * d3[Qcols[i]]
        end
        Qx = mul_Qx_COO!(Qx, Qrows, Qcols, Qvals, x)
        xTQx_2 =  x' * Qx / 2
        for i=1:length(Arows)
            Avals[i] /= d1[Arows[i]] * d2[Acols[i]] * d3[Acols[i]]
        end
        λ .*= d1
        ATλ = mul_ATλ_COO!(ATλ, Arows, Acols, Avals, λ)
        Ax = mul_Ax_COO!(Ax, Arows, Acols, Avals, x)
        b ./= d1
        c ./= d2 .* d3
        cTx = c' * x
        pri_obj = xTQx_2 + cTx + c0
        lvar .*= d2 .* d3
        uvar .*= d2 .* d3
        dual_obj = b' * λ - xTQx_2 + view(s_l,ilow)'*view(lvar,ilow) -
                    view(s_u,iupp)'*view(uvar,iupp) +c0

        s_l ./= d2 .* d3
        s_u ./= d2 .* d3
        rb .= Ax .- b
        rc .= ATλ .-Qx .+ s_l .- s_u .- c
#         rcNorm, rbNorm = norm(rc), norm(rb)
        rcNorm, rbNorm = norm(rc, Inf), norm(rb, Inf)
    end

    elapsed_time = time() - start_time

    stats = GenericExecutionStats(status, QM, solution = x[1:QM.meta.nvar],
                                  objective = pri_obj ,
                                  dual_feas = rcNorm,
                                  primal_feas = rbNorm,
                                  multipliers = λ,
                                  multipliers_L = s_l,
                                  multipliers_U = s_u,
                                  iter = k,
                                  elapsed_time=elapsed_time)
    return stats
end

end
