mutable struct QM_FloatData_ref{T<:Real} <: Abstract_QM_FloatData{T}
    Q                   :: SparseMatrixCSC{T,Int}
    AT                  :: SparseMatrixCSC{T,Int} # using AT is easier to form systems
    b                   :: Vector{T}
    c                   :: Vector{T}
    c0                  :: T
    lvar                :: Vector{T}
    uvar                :: Vector{T}
    Δref                :: T
    x_approx            :: Vector{T}
    x_approxQx_approx_2 :: T
    b_init              :: Vector{T}
    bTy_approx          :: T
    c_init              :: Vector{T}
    pri_obj_approx      :: T
end

function fd_refinement(fd :: QM_FloatData{T}, id :: QM_IntData, res :: residuals{T}, Δxy :: Vector{T}, pt :: point{T}, 
                       itd :: iter_data{T}, ϵ :: tolerances{T}, pad :: preallocated_data{T}, cnts :: counters, T0 :: DataType,
                       refinement :: Symbol; centering :: Bool = false) where {T<:Real}

    # center points before zoom
    if centering
        if itd.fact_fail # re-factorize if it failed in a lower precision system
            out = factorize_K2!(itd.J_augm, itd.J_fact, itd.tmp_diag, itd.diag_Q, itd.diagind_J, itd.regu, 
                                pt.s_l, pt.s_u, itd.x_m_lvar, itd.uvar_m_x, id.ilow, id.iupp, 
                                id.n_rows, id.n_cols, cnts, itd.qp, T, T0)
        end
        pad.rxs_l .= -itd.μ 
        pad.rxs_u .= itd.μ 
        solve_augmented_system_cc!(itd.J_fact, pad.Δxy, pad.Δs_l, pad.Δs_u, itd.x_m_lvar, itd.uvar_m_x, 
                                pad.rxs_l, pad.rxs_u, pt.s_l, pt.s_u, id.ilow, id.iupp)
        α_pri, α_dual = compute_αs(pt.x, pt.s_l, pt.s_u, fd.lvar, fd.uvar, pad.Δxy, pad.Δs_l, pad.Δs_u, id.n_cols)
        update_data!(pt, α_pri, α_dual, itd, pad, res, fd, id)
    end

    c_ref = fd.c .- itd.ATy .+ itd.Qx
    if refinement == :zoom || refinement == :multizoom
        # zoom parameter
        Δref = one(T) / res.rbNorm
    elseif refinement == :ref || refinement == :multiref
        Δref = one(T)
        αref = T(1.0e12)
        δd = norm(c_ref, Inf) 
        if id.n_low == 0 && id.n_upp > 0
            δp  = max(res.rbNorm, maximum(itd.uvar_m_x))
        elseif id.n_low > 0 && id.n_upp == 0
            δp  = max(res.rbNorm, maximum(itd.x_m_lvar))
        elseif id.n_low == 0 && id.n_upp == 0
            δp  = max(res.rbNorm)
        else
            δp  = max(res.rbNorm, maximum(itd.x_m_lvar), maximum(itd.uvar_m_x))
        end
        Δref = max(αref / Δref, one(T) / δp, one(T) / δd)
    end

    c_ref .*= Δref
    fd_ref = QM_FloatData_ref(fd.Q, fd.AT, .-res.rb .* Δref, c_ref, fd.c0, 
                             Δref .* (fd.lvar .- pt.x), Δref .* (fd.uvar .- pt.x), Δref, pt.x, copy(itd.xTQx_2), fd.b, dot(fd.b, pt.y),
                             fd.c, copy(itd.pri_obj))

    # init zoom points
    pt_z = point(zeros(T, id.n_cols), zeros(T, id.n_rows), max.(abs.(c_ref[id.ilow]), eps(T)), max.(abs.(c_ref[id.iupp]), eps(T)))
    starting_points!(pt_z, fd_ref, id, itd)

    # update residuals
    res.rb .= itd.Ax .- fd_ref.b
    res.rc .= itd.ATy .- itd.Qx .- fd_ref.c
    res.rc[id.ilow] .+= pt_z.s_l
    res.rc[id.iupp] .-= pt_z.s_u
    res.rcNorm, res.rbNorm = norm(res.rc, Inf), norm(res.rb, Inf)

    return fd_ref, pt_z
end

function update_pt_ref!(Δref :: T, pt :: point{T}, pt_z :: point{T}, res :: residuals{T}, id :: QM_IntData,
                        fd :: QM_FloatData{T}, itd :: iter_data{T}) where {T<:Real}
    
    # update point                    
    pt.x .+= pt_z.x ./ Δref
    pt.y .+= pt_z.y ./ Δref
    pt.s_l .= pt_z.s_l ./ Δref
    pt.s_u .= pt_z.s_u ./ Δref

    # update iter_data
    itd.Qx = mul!(itd.Qx, Symmetric(fd.Q, :U), pt.x)
    itd.xTQx_2 = dot(pt.x, itd.Qx) / 2
    itd.ATy = mul!(itd.ATy, fd.AT, pt.y)
    itd.Ax = mul!(itd.Ax, fd.AT', pt.x)
    itd.cTx = dot(fd.c, pt.x)
    itd.pri_obj = itd.xTQx_2 + itd.cTx + fd.c0
    itd.dual_obj = @views dot(fd.b, pt.y) - itd.xTQx_2 + dot(pt.s_l, fd.lvar[id.ilow]) - dot(pt.s_u, fd.uvar[id.iupp]) + fd.c0  
    itd.pdd = abs(itd.pri_obj - itd.dual_obj ) / (one(T) + abs(itd.pri_obj)) 

    # update residuals
    res.rb .= itd.Ax .- fd.b
    res.rc .= itd.ATy .- itd.Qx .- fd.c
    res.rc[id.ilow] .+= pt.s_l
    res.rc[id.iupp] .-= pt.s_u
    res.rcNorm, res.rbNorm = norm(res.rc, Inf), norm(res.rb, Inf)
    res.rcNorm, res.rbNorm = norm(res.rc, Inf), norm(res.rb, Inf)
end



function update_data!(pt :: point{T}, α_pri :: T, α_dual :: T, itd :: iter_data{T}, pad :: preallocated_data{T}, 
                      res :: residuals{T}, fd :: QM_FloatData_ref{T}, id :: QM_IntData) where {T<:Real}

    # (x, y, s_l, s_u) += α * Δ
    update_pt!(pt.x, pt.y, pt.s_l, pt.s_u, α_pri, α_dual, pad.Δxy, pad.Δs_l, pad.Δs_u, id.n_rows, id.n_cols)

    # update iter_data
    itd.x_m_lvar .= @views pt.x[id.ilow] .- fd.lvar[id.ilow]
    itd.uvar_m_x .= @views fd.uvar[id.iupp] .- pt.x[id.iupp]
    boundary_safety!(itd.x_m_lvar, itd.uvar_m_x, id.n_low, id.n_upp, T)

    itd.μ = compute_μ(itd.x_m_lvar, itd.uvar_m_x, pt.s_l, pt.s_u, id.n_low, id.n_upp)
    itd.Qx = mul!(itd.Qx, Symmetric(fd.Q, :U), pt.x)
    x_approxQx = dot(fd.x_approx, itd.Qx)
    itd.xTQx_2 =  dot(pt.x, itd.Qx) / 2
    itd.ATy = mul!(itd.ATy, fd.AT, pt.y)
    itd.Ax = mul!(itd.Ax, fd.AT', pt.x)
    itd.cTx = dot(fd.c_init, pt.x)
    itd.pri_obj = fd.pri_obj_approx + x_approxQx / fd.Δref + itd.xTQx_2 / fd.Δref^2 + dot(fd.c_init, pt.x) / fd.Δref
    itd.dual_obj = @views fd.bTy_approx + dot(fd.b_init, pt.y) / fd.Δref - fd.x_approxQx_approx_2 - dot(fd.x_approx, itd.Qx) / fd.Δref - 
                        itd.xTQx_2 / fd.Δref^2 +
                        dot(pt.s_l, fd.lvar[id.ilow]) / fd.Δref^2 + dot(pt.s_l, fd.x_approx[id.ilow]) / fd.Δref - 
                        dot(pt.s_u, fd.uvar[id.iupp]) / fd.Δref^2 - dot(pt.s_u, fd.x_approx[id.iupp]) / fd.Δref + fd.c0  
    itd.pdd = abs(itd.pri_obj - itd.dual_obj ) / (one(T) + abs(itd.pri_obj)) 

    #update residuals
    res.n_Δx = @views α_pri * norm(pad.Δxy[1:id.n_cols])
    res.rb .= itd.Ax .- fd.b
    res.rc .= itd.ATy .- itd.Qx .- fd.c
    res.rc[id.ilow] .+= pt.s_l
    res.rc[id.iupp] .-= pt.s_u
    res.rcNorm, res.rbNorm = norm(res.rc, Inf) / fd.Δref, norm(res.rb, Inf) / fd.Δref
end