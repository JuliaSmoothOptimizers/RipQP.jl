function starting_points(fd :: QM_FloatData{T}, id:: QM_IntData, itd :: iter_data{T},
                         Δ_xy :: Vector{T}) where {T<:Real}

    # solve [-Q-tmp_diag    A' ] [x] = [b]  to initialize (x, y, s_l, s_u)
    #       [      A         0 ] [y] = [0]
    Δ_xy[id.n_cols+1: end] = fd.b
    Δ_xy = ldiv!(itd.J_fact, Δ_xy)
    pt0 = point(Δ_xy[1:id.n_cols], Δ_xy[id.n_cols+1:end], zeros(T, id.n_cols), zeros(T, id.n_cols))
    itd.Qx = mul!(itd.Qx, Symmetric(fd.Q, :U), pt0.x)
    itd.ATy = mul!(itd.ATy, fd.AT, pt0.y)
    dual_val = itd.Qx .- itd.ATy .+ fd.c
    pt0.s_l[id.ilow] = @views dual_val[id.ilow]
    pt0.s_u[id.iupp] = @views -dual_val[id.iupp]

    # check distance to bounds δ for x, s_l and s_u
    itd.x_m_lvar .= @views pt0.x[id.ilow] .- fd.lvar[id.ilow]
    itd.uvar_m_x .= @views fd.uvar[id.iupp] .- pt0.x[id.iupp]
    if id.n_low == 0
        δx_l1, δs_l1 = zero(T), zero(T)
    else
        δx_l1 = max(-T(1.5)*minimum(itd.x_m_lvar), T(1.e-2))
        δs_l1 = @views max(-T(1.5)*minimum(pt0.s_l[id.ilow]), T(1.e-4))
    end
    if id.n_upp == 0
        δx_u1, δs_u1 = zero(T), zero(T)
    else
        δx_u1 = max(-T(1.5)*minimum(itd.uvar_m_x), T(1.e-2))
        δs_u1 = @views max(-T(1.5)*minimum(pt0.s_u[id.iupp]), T(1.e-4))
    end

    # correct components that to not respect the bounds 
    itd.x_m_lvar .+= δx_l1
    itd.uvar_m_x .+= δx_u1
    s0_l1 = @views pt0.s_l[id.ilow] .+ δs_l1
    s0_u1 = @views pt0.s_u[id.iupp] .+ δs_u1
    xs_l1, xs_u1 = s0_l1' * itd.x_m_lvar, s0_u1' * itd.uvar_m_x
    if id.n_low == 0
        δx_l2, δs_l2 = zero(T), zero(T)
    else
        δx_l2 = δx_l1 + xs_l1 / sum(s0_l1) / 2
        δs_l2 = @views δs_l1 + xs_l1 / sum(itd.x_m_lvar) / 2
    end
    if id.n_upp == 0
        δx_u2, δs_u2 = zero(T), zero(T)
    else
        δx_u2 = δx_u1 + xs_u1 / sum(s0_u1) / 2
        δs_u2 = @views δs_u1 + xs_u1 / sum(itd.uvar_m_x) / 2
    end
    δx = max(δx_l2, δx_u2)
    δs = max(δs_l2, δs_u2)
    pt0.x[id.ilow] .+= δx
    pt0.x[id.iupp] .-= δx
    pt0.s_l[id.ilow] .= @views pt0.s_l[id.ilow] .+ δs
    pt0.s_u[id.iupp] .= @views pt0.s_u[id.iupp] .+ δs

    # deal with the compensation phaenomenon in x if irng != []
    @inbounds @simd for i in id.irng
        if fd.lvar[i] >= pt0.x[i]
            pt0.x[i] = fd.lvar[i] + T(1e-4)
        end
        if pt0.x[i] >= fd.uvar[i]
            pt0.x[i] = fd.uvar[i] - T(1e-4)
        end
        if (fd.lvar[i] < pt0.x[i] < fd.uvar[i]) == false
            pt0.x[i] = (fd.lvar[i] + fd.uvar[i]) / 2
        end
    end

    # verify bounds 
    @assert all(pt0.x .> fd.lvar) && all(pt0.x .< fd.uvar)
    @assert @views all(pt0.s_l[id.ilow] .> zero(T)) && all(pt0.s_u[id.iupp] .> zero(T))

    # update itd
    update_iter_data!(itd, pt0, fd, id; safety = false)

    return pt0, itd, Δ_xy
end
