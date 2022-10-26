# tools for the regularization of the system.

# update regularization values in classic mode if there is a failure during factorization
function update_regu_trycatch!(regu::Regularization{T}, cnts::Counters) where {T}
  !cnts.last_sp && return 1
  if cnts.c_regu_dim == 0 && cnts.c_catch == 0
    regu.δ *= T(1e2)
    regu.δ_min *= T(1e2)
    regu.ρ *= T(1e3)
    regu.ρ_min *= T(1e3)
  elseif cnts.c_regu_dim == 0 && cnts.c_catch != 0
    regu.δ *= T(1e1)
    regu.δ_min *= T(1e1)
    regu.ρ *= T(1e0)
    regu.ρ_min *= T(1e0)
  elseif cnts.c_regu_dim != 0 && cnts.c_catch == 0
    regu.δ *= T(1e5)
    regu.δ_min *= T(1e5)
    regu.ρ *= T(1e5)
    regu.ρ_min *= T(1e5)
  else
    regu.δ *= T(1e1)
    regu.δ_min *= T(1e1)
    regu.ρ *= T(1e1)
    regu.ρ_min *= T(1e1)
  end
  return 0
end

function update_regu!(regu)
  if regu.δ >= regu.δ_min
    regu.δ /= 10
  end
  if regu.ρ >= regu.ρ_min
    regu.ρ /= 10
  end
end

# update regularization, and corrects if the magnitude of the diagonal of the matrix is too high
function update_regu_diagK2!(
  regu::Regularization{T},
  K::Symmetric{<:Real, <:SparseMatrixCSC},
  diagind_K,
  μ::T,
  nvar::Int,
  cnts::Counters;
  safety_dist_bnd::Bool = true,
) where {T}
  update_regu_diagK2!(
    regu,
    K.data.nzval,
    diagind_K,
    μ,
    nvar,
    cnts,
    safety_dist_bnd,
  )
end

function update_regu_diagK2!(
  regu::Regularization{T},
  K::Symmetric{<:Real, <:SparseMatrixCOO},
  diagind_K,
  μ,
  nvar::Int,
  cnts::Counters;
  safety_dist_bnd::Bool = false,
) where {T}
  update_regu_diagK2!(
    regu,
    K.data.vals,
    diagind_K,
    μ,
    nvar,
    cnts,
    safety_dist_bnd,
  )
end

function update_regu_diagK2!(
  regu::Regularization{T},
  K_nzval::AbstractVector{T},
  diagind_K,
  μ::T,
  nvar::Int,
  cnts::Counters,
  safety_dist_bnd::Bool,
) where {T}
  if safety_dist_bnd
    if T == Float64 && regu.regul == :classic && cnts.k > 10 && μ ≤ eps(T) && cnts.c_regu_dim < 20
      regu.δ_min /= 10
      regu.δ /= 10
      cnts.c_regu_dim += 1
    end
    if T == Float64 &&
       regu.regul == :classic &&
       cnts.k > 10 &&
       cnts.c_catch <= 1 &&
       regu.δ_min >= eps(T)^(4 / 5) &&
       @views minimum(K_nzval[diagind_K[1:nvar]]) < -one(T) / regu.δ / T(1e-6)
      regu.δ /= 10
      regu.δ_min /= 10
      cnts.c_regu_dim += 1
    elseif !cnts.last_sp &&
           cnts.c_regu_dim <= 2 &&
           cnts.k ≥ 5 &&
           @views minimum(K_nzval[diagind_K[1:nvar]]) < -one(T) / eps(T) &&
                  @views maximum(K_nzval[diagind_K[1:nvar]]) > -one(T) / 10
      regu.regul == :classic && return 1
    elseif T == Float128 &&
           cnts.k > 10 &&
           cnts.c_catch <= 1 &&
           @views minimum(K_nzval[diagind_K[1:nvar]]) < -one(T) / regu.δ / T(1e-15)
      regu.δ /= 10
      regu.δ_min /= 10
      cnts.c_regu_dim += 1
    end
  end

  update_regu!(regu)

  return 0
end

function update_regu_diagK2_5!(
  regu::Regularization{T},
  D::AbstractVector{T},
  μ::T,
  cnts::Counters,
) where {T}
  if T == Float64 && cnts.k > 10 && μ ≤ eps(T) && cnts.c_regu_dim < 5
    regu.δ_min /= 10
    regu.δ /= 10
    cnts.c_regu_dim += 1
  end
  if T == Float64 &&
     cnts.k > 10 &&
     cnts.c_catch <= 1 &&
     @views minimum(D) < -one(T) / regu.δ / T(1e-6)
    regu.δ /= 10
    regu.δ_min /= 10
    cnts.c_regu_dim += 1
  elseif !cnts.last_sp && cnts.c_regu_dim < 2 && @views minimum(D) < -one(T) / regu.δ / T(1e-5)
    return 1
  elseif T == Float128 &&
         cnts.k > 10 &&
         cnts.c_catch <= 1 &&
         @views minimum(D) < -one(T) / regu.δ / T(1e-15)
    regu.δ /= 10
    regu.δ_min /= 10
    cnts.c_regu_dim += 1
  end

  update_regu!(regu)

  return 0
end
