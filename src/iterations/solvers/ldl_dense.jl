function ldl_dense!(A :: Symmetric{T, Matrix{T}}) where {T <: Real}
  # use symmetric lower only
  @assert A.uplo == 'L'
  n = size(A)[1]
  for j=1:n
      djj = A[j,j]
      for k=1:(j-1)
          djj -= A[j,k]^2 * A[k,k]
      end
      A[j,j] = djj
      for i=(j+1):n
          lij = A[i,j] # we use the lower block
          for k=1:(j-1)
              lij -= A[i,k] * A[j,k] * A[k,k]
          end
          A.data[i,j] = lij / A[j,j]
      end
  end
end

function solve_lowertri!(L,b)
  n = length(b)
  for j=1:n
      for i=j+1:n
          b[i] -= L[i,j] * b[j] 
      end 
  end
end

function solve_lowertri_transpose!(LT, b)
  n = length(b)
  for i=n:-1:1
      for j=(i-1):-1:1
          b[j] -= conj(LT[i,j]) * b[i] 
      end 
  end
end

function solve_diag!(D, b)
  n = length(b)
  for j=1:n
      b[j] /= D[j,j]
  end
end

# function solve(L, D, b)
#   x = copy(b)
#   solve_lowertri!(L, x)
#   solve_diag!(D, x)
#   solve_lowertri_transpose!(L, x)
#   return x
# end

function ldiv_dense!(LD, b)
  solve_lowertri!(LD, b)
  @views solve_diag!(Diagonal(LD[diagind(LD)]), b)
  solve_lowertri_transpose!(LD, b)
  return b
end

function ldiv_dense(x, LD, b)
  x .= b
  ldiv_dense!(LD, x)
end