# LinearAlgebra's sparse function that returns the colptr, rowval and nzval vectors from a sparse COO format.
# The original code is here: 
# https://github.com/JuliaLang/julia/blob/1b93d53fc4bb59350ada898038ed4de2994cce33/stdlib/SparseArrays/src/sparsematrix.jl

function sparse_coords(
  I::AbstractVector{Ti},
  J::AbstractVector{Ti},
  V::AbstractVector{Tv},
  m::Integer,
  n::Integer,
  combine,
) where {Tv, Ti <: Integer}
  SparseArrays.require_one_based_indexing(I, J, V)
  coolen = length(I)
  if length(J) != coolen || length(V) != coolen
    throw(
      ArgumentError(
        string(
          "the first three arguments' lengths must match, ",
          "length(I) (=$(length(I))) == length(J) (= $(length(J))) == length(V) (= ",
          "$(length(V)))",
        ),
      ),
    )
  end
  if Base.hastypemax(Ti) && coolen >= typemax(Ti)
    throw(ArgumentError("the index type $Ti cannot hold $coolen elements; use a larger index type"))
  end
  if m == 0 || n == 0 || coolen == 0
    if coolen != 0
      if n == 0
        throw(ArgumentError("column indices J[k] must satisfy 1 <= J[k] <= n"))
      elseif m == 0
        throw(ArgumentError("row indices I[k] must satisfy 1 <= I[k] <= m"))
      end
    end
    return m, n, fill(one(Ti), n + 1), Vector{Ti}(), Vector{Tv}()
  else
    # Allocate storage for CSR form
    csrrowptr = Vector{Ti}(undef, m + 1)
    csrcolval = Vector{Ti}(undef, coolen)
    csrnzval = Vector{Tv}(undef, coolen)

    # Allocate storage for the CSC form's column pointers and a necessary workspace
    csccolptr = Vector{Ti}(undef, n + 1)
    klasttouch = Vector{Ti}(undef, n)

    # Allocate empty arrays for the CSC form's row and nonzero value arrays
    # The parent method called below automagically resizes these arrays
    cscrowval = Vector{Ti}()
    cscnzval = Vector{Tv}()

    sparse_coords!(
      I,
      J,
      V,
      m,
      n,
      combine,
      klasttouch,
      csrrowptr,
      csrcolval,
      csrnzval,
      csccolptr,
      cscrowval,
      cscnzval,
    )
  end
end

function sparse_coords!(
  I::AbstractVector{Ti},
  J::AbstractVector{Ti},
  V::AbstractVector{Tv},
  m::Integer,
  n::Integer,
  combine,
  klasttouch::Vector{Tj},
  csrrowptr::Vector{Tj},
  csrcolval::Vector{Ti},
  csrnzval::Vector{Tv},
  csccolptr::Vector{Ti},
  cscrowval::Vector{Ti},
  cscnzval::Vector{Tv},
) where {Tv, Ti <: Integer, Tj <: Integer}
  SparseArrays.require_one_based_indexing(I, J, V)
  SparseArrays.sparse_check_Ti(m, n, Ti)
  SparseArrays.sparse_check_length("I", I, 0, Tj)
  # Compute the CSR form's row counts and store them shifted forward by one in csrrowptr
  fill!(csrrowptr, Tj(0))
  coolen = length(I)
  min(length(J), length(V)) >= coolen ||
    throw(ArgumentError("J and V need length >= length(I) = $coolen"))
  @inbounds for k = 1:coolen
    Ik = I[k]
    if 1 > Ik || m < Ik
      throw(ArgumentError("row indices I[k] must satisfy 1 <= I[k] <= m"))
    end
    csrrowptr[Ik + 1] += Tj(1)
  end

  # Compute the CSR form's rowptrs and store them shifted forward by one in csrrowptr
  countsum = Tj(1)
  csrrowptr[1] = Tj(1)
  @inbounds for i = 2:(m + 1)
    overwritten = csrrowptr[i]
    csrrowptr[i] = countsum
    countsum += overwritten
  end

  # Counting-sort the column and nonzero values from J and V into csrcolval and csrnzval
  # Tracking write positions in csrrowptr corrects the row pointers
  @inbounds for k = 1:coolen
    Ik, Jk = I[k], J[k]
    if Ti(1) > Jk || Ti(n) < Jk
      throw(ArgumentError("column indices J[k] must satisfy 1 <= J[k] <= n"))
    end
    csrk = csrrowptr[Ik + 1]
    @assert csrk >= Tj(1) "index into csrcolval exceeds typemax(Ti)"
    csrrowptr[Ik + 1] = csrk + Tj(1)
    csrcolval[csrk] = Jk
    csrnzval[csrk] = V[k]
  end
  # This completes the unsorted-row, has-repeats CSR form's construction

  # Sweep through the CSR form, simultaneously (1) calculating the CSC form's column
  # counts and storing them shifted forward by one in csccolptr; (2) detecting repeated
  # entries; and (3) repacking the CSR form with the repeated entries combined.
  #
  # Minimizing extraneous communication and nonlocality of reference, primarily by using
  # only a single auxiliary array in this step, is the key to this method's performance.
  fill!(csccolptr, Ti(0))
  fill!(klasttouch, Tj(0))
  writek = Tj(1)
  newcsrrowptri = Ti(1)
  origcsrrowptri = Tj(1)
  origcsrrowptrip1 = csrrowptr[2]
  @inbounds for i = 1:m
    for readk = origcsrrowptri:(origcsrrowptrip1 - Tj(1))
      j = csrcolval[readk]
      if klasttouch[j] < newcsrrowptri
        klasttouch[j] = writek
        if writek != readk
          csrcolval[writek] = j
          csrnzval[writek] = csrnzval[readk]
        end
        writek += Tj(1)
        csccolptr[j + 1] += Ti(1)
      else
        klt = klasttouch[j]
        csrnzval[klt] = combine(csrnzval[klt], csrnzval[readk])
      end
    end
    newcsrrowptri = writek
    origcsrrowptri = origcsrrowptrip1
    origcsrrowptrip1 != writek && (csrrowptr[i + 1] = writek)
    i < m && (origcsrrowptrip1 = csrrowptr[i + 2])
  end

  # Compute the CSC form's colptrs and store them shifted forward by one in csccolptr
  countsum = Tj(1)
  csccolptr[1] = Ti(1)
  @inbounds for j = 2:(n + 1)
    overwritten = csccolptr[j]
    csccolptr[j] = countsum
    countsum += overwritten
    Base.hastypemax(Ti) && (
      countsum <= typemax(Ti) ||
      throw(ArgumentError("more than typemax(Ti)-1 == $(typemax(Ti)-1) entries"))
    )
  end

  # Now knowing the CSC form's entry count, resize cscrowval and cscnzval if necessary
  cscnnz = countsum - Tj(1)
  length(cscrowval) < cscnnz && resize!(cscrowval, cscnnz)
  length(cscnzval) < cscnnz && resize!(cscnzval, cscnnz)

  # Finally counting-sort the row and nonzero values from the CSR form into cscrowval and
  # cscnzval. Tracking write positions in csccolptr corrects the column pointers.
  @inbounds for i = 1:m
    for csrk = csrrowptr[i]:(csrrowptr[i + 1] - Tj(1))
      j = csrcolval[csrk]
      x = csrnzval[csrk]
      csck = csccolptr[j + 1]
      csccolptr[j + 1] = csck + Ti(1)
      cscrowval[csck] = i
      cscnzval[csck] = x
    end
  end

  return m, n, csccolptr, cscrowval, cscnzval
end
function sparse_coords!(
  I::AbstractVector{Ti},
  J::AbstractVector{Ti},
  V::AbstractVector{Tv},
  m::Integer,
  n::Integer,
  combine,
  klasttouch::Vector{Tj},
  csrrowptr::Vector{Tj},
  csrcolval::Vector{Ti},
  csrnzval::Vector{Tv},
  csccolptr::Vector{Ti},
) where {Tv, Ti <: Integer, Tj <: Integer}
  sparse_coords!(
    I,
    J,
    V,
    m,
    n,
    combine,
    klasttouch,
    csrrowptr,
    csrcolval,
    csrnzval,
    csccolptr,
    Vector{Ti}(),
    Vector{Tv}(),
  )
end
function sparse_coords!(
  I::AbstractVector{Ti},
  J::AbstractVector{Ti},
  V::AbstractVector{Tv},
  m::Integer,
  n::Integer,
  combine,
  klasttouch::Vector{Tj},
  csrrowptr::Vector{Tj},
  csrcolval::Vector{Ti},
  csrnzval::Vector{Tv},
) where {Tv, Ti <: Integer, Tj <: Integer}
  sparse_coords!(
    I,
    J,
    V,
    m,
    n,
    combine,
    klasttouch,
    csrrowptr,
    csrcolval,
    csrnzval,
    Vector{Ti}(undef, n + 1),
    Vector{Ti}(),
    Vector{Tv}(),
  )
end

sparse_coords(
  I::AbstractVector,
  J::AbstractVector,
  V::AbstractVector,
  m::Integer,
  n::Integer,
  combine,
) = sparse_coords(AbstractVector{Int}(I), AbstractVector{Int}(J), V, m, n, combine)

sparse_coords(I, J, V::AbstractVector) =
  sparse_coords(I, J, V, SparseArrays.dimlub(I), SparseArrays.dimlub(J))

sparse_coords(I, J, V::AbstractVector, m, n) = sparse_coords(I, J, V, Int(m), Int(n), +)
