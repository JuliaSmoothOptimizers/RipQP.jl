function write_system(
  w::SystemWrite,
  K::SparseMatrixCSC{T, Int},
  rhs::Vector{T},
  step::Symbol,
  iter::Int,
) where {T <: Real}
  if rem(iter + 1 - w.kfirst, w.kgap) == 0
    if step != :cc
      K_str = string(w.name, "K_iter", iter + 1, ".mtx")
      MatrixMarket.mmwrite(K_str, K)
    end
    rhs_str = string(w.name, "rhs_iter", iter + 1, "_", step, ".rhs")
    open(rhs_str, "w") do io
      writedlm(io, rhs)
    end
  end
end
