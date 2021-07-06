export K2minresParams

"""
Type to use the K2 formulation with MINRES, using the package 
[`Krylov.jl`](https://github.com/JuliaSmoothOptimizers/Krylov.jl). 
The outer constructor 
    K2minresParams(; preconditioner = :Jacobi, ratol = 1.0e-10, rrtol = 1.0e-10)
creates a [`RipQP.SolverParams`](@ref) that should be used to create a [`RipQP.InputConfig`](@ref).
The list of available preconditionners for this solver is displayed here: [`RipQP.PreconditionerDataK2`](@ref)
"""
struct K2minresParams <: SolverParams
    preconditioner  :: Symbol
    ratol           :: Float64
    rrtol           :: Float64
end

function K2minresParams(; preconditioner = :Jacobi, ratol :: T = 1.0e-10, rrtol :: T = 1.0e-10) where {T<:Real} 
    return K2minresParams(preconditioner, ratol, rrtol)
end

mutable struct PreallocatedData_K2minres{T<:Real, S, Fv, Fu, Fw} <: PreallocatedData{T, S} 
    pdat             :: PreconditionerDataK2{T, S}
    D                :: S                                  # temporary top-left diagonal
    rhs              :: S
    regu             :: Regularization{T}
    δv               :: S
    K                :: LinearOperator{T, Fv, Fu, Fw} # augmented matrix          
    MS               :: MinresSolver{T, S}
    ratol            :: T
    rrtol            :: T
end

function opK2prod!(res::AbstractVector{T}, nvar::Int, Q::AbstractMatrix{T}, D::AbstractVector{T}, 
                  AT::AbstractMatrix{T}, δv::AbstractVector{T}, v::AbstractVector{T}, α::T, β::T) where T
  @views mul!(res[1: nvar], Q, v[1: nvar], -α, β)
  res[1: nvar] .+= α .* D .* v[1: nvar]
  @views mul!(res[1: nvar], AT, v[nvar+1: end], α, one(T))
  @views mul!(res[nvar+1: end], AT', v[1: nvar], α, β)
  res[nvar+1: end] .+= @views (α * δv[1]) .* v[nvar+1: end]
end

function PreallocatedData(sp :: K2minresParams, fd :: QM_FloatData{T}, id :: QM_IntData, 
                          iconf :: InputConfig{Tconf}) where {T<:Real, Tconf<:Real}

  # init Regularization values
  D = similar(fd.c, id.nvar)
  if iconf.mode == :mono
    regu = Regularization(
      T(sqrt(eps()) * 1e5),
      T(sqrt(eps()) * 1e5),
      1e2 * sqrt(eps(T)),
      1e2 * sqrt(eps(T)),
      :classic,
    )
    D .= -T(1.0e0) / 2
  else
    regu = Regularization(
      T(sqrt(eps()) * 1e5),
      T(sqrt(eps()) * 1e5),
      T(sqrt(eps(T)) * 1e0),
      T(sqrt(eps(T)) * 1e0),
      :classic,
    )
    D .= -T(1.0e-2)
  end
  δv = [regu.δ]
  K = LinearOperator(T, id.nvar + id.ncon, id.nvar + id.ncon, true, true, 
                     (res, v, α, β) -> opK2prod!(res, id.nvar, fd.Q, D, fd.AT, δv, v, α, β))

  rhs = similar(fd.c, id.nvar+id.ncon)
  MS = MinresSolver(K, rhs)

  pdat = eval(sp.preconditioner)(id, fd, regu, D, K)

  return PreallocatedData_K2minres(pdat,
                                   D,
                                   rhs, 
                                   regu,
                                   δv,
                                   K, #K
                                   MS, #K_fact
                                   sp.ratol,
                                   sp.rrtol,
                                   )
end

function solver!(pad :: PreallocatedData_K2minres{T}, dda :: DescentDirectionAllocs{T}, pt :: Point{T}, itd :: IterData{T}, 
                 fd :: Abstract_QM_FloatData{T}, id :: QM_IntData, res :: Residuals{T}, cnts :: Counters, T0 :: DataType, 
                 step :: Symbol) where {T<:Real} 

  # erase dda.Δxy_aff only for affine predictor step with PC method
  pad.rhs .= (step == :aff) ? dda.Δxy_aff : pad.rhs .= itd.Δxy
  rhsNorm = norm(pad.rhs)
  if rhsNorm != zero(T)
    pad.rhs ./= rhsNorm
  end

  # pop = pad.pdat.P*pad.K
  # M = Matrix(pop)
  # println(M[diagind(M)])
  (pad.MS.x, pad.MS.stats) = minres!(pad.MS, pad.K, pad.rhs, M=pad.pdat.P, 
                                     verbose=0, atol=zero(T), rtol=zero(T), ratol=pad.ratol, rrtol=pad.rrtol)
  if rhsNorm != zero(T)
    pad.MS.x .*= rhsNorm
  end

  if step == :aff 
    dda.Δxy_aff .= pad.MS.x
  else
    itd.Δxy .= pad.MS.x
  end

  return 0
end

function update_pad!(pad :: PreallocatedData_K2minres{T}, dda :: DescentDirectionAllocs{T}, pt :: Point{T}, itd :: IterData{T}, 
                     fd :: Abstract_QM_FloatData{T}, id :: QM_IntData, res :: Residuals{T}, cnts :: Counters, T0 :: DataType) where {T<:Real}

    if cnts.k != 0
        update_regu!(pad.regu) 
    end

    pad.D .= -pad.regu.ρ
    pad.D[id.ilow] .-= pt.s_l ./ itd.x_m_lvar
    pad.D[id.iupp] .-= pt.s_u ./ itd.uvar_m_x
    pad.δv[1] = pad.regu.δ

    update_preconditioner!(pad.pdat, pad, itd, pt, id, fd, cnts)

    return 0
end
