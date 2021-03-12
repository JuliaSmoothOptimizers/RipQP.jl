function convert_FloatData(T :: DataType, fd_T0 :: QM_FloatData{T0}) where {T0<:Real}
    return QM_FloatData(SparseMatrixCSC{T, Int}(fd_T0.Q.m, fd_T0.Q.n, 
                                                fd_T0.Q.colptr, fd_T0.Q.rowval, Array{T}(fd_T0.Q.nzval)),
                        SparseMatrixCSC{T, Int}(fd_T0.AT.m, fd_T0.AT.n, 
                                                fd_T0.AT.colptr, fd_T0.AT.rowval, Array{T}(fd_T0.AT.nzval)),
                        Array{T}(fd_T0.b), 
                        Array{T}(fd_T0.c), 
                        T(fd_T0.c0),
                        Array{T}(fd_T0.lvar), 
                        Array{T}(fd_T0.uvar))
end

function convert_types(T :: DataType, pt :: point{T_old}, itd :: iter_data{T_old}, res :: residuals{T_old},
                       pad :: preallocated_data{T_old}, T0 :: DataType) where {T_old<:Real}

   pt = convert(point{T}, pt)
   res = convert(residuals{T}, res)
   itd = convert(iter_data{T}, itd)
   if T == Float64 && T0 == Float64
       itd.regu.ρ_min, itd.regu.δ_min = T(sqrt(eps())*1e-5), T(sqrt(eps())*1e0)
   else
       itd.regu.ρ_min, itd.regu.δ_min = T(sqrt(eps(T))*1e1), T(sqrt(eps(T))*1e1)
   end
   pad = convert(preallocated_data{T}, pad)

   itd.regu.ρ /= 10
   itd.regu.δ /= 10

   return pt, itd, res, pad
end

function iter_and_update_T!(pt :: point{T}, itd :: iter_data{T}, fd_T :: Abstract_QM_FloatData{T}, id :: QM_IntData, res :: residuals{T}, 
                            sc :: stop_crit{Tsc}, pad :: preallocated_data{T}, ϵ_T :: tolerances{T}, ϵ :: tolerances{T0}, solve! :: Function, 
                            cnts :: counters, max_iter_T :: Int, T_next :: DataType, display :: Bool) where {T<:Real, T0<:Real, Tsc<:Real}
    # iters T
    sc.max_iter = max_iter_T
    iter!(pt, itd, fd_T, id, res, sc, pad, ϵ_T, solve!, cnts, T0, display)

    # convert to T_next
    pt, itd, res, pad = convert_types(T_next, pt, itd, res, pad, T0)
    sc.optimal = itd.pdd < ϵ.pdd && res.rbNorm < ϵ.tol_rb && res.rcNorm < ϵ.tol_rc
    sc.small_Δx, sc.small_μ = res.n_Δx < ϵ.Δx, itd.μ < ϵ.μ
    return pt, itd, res, pad 
end