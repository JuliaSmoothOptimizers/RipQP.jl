include("augmented/augmented.jl")
include("Newton/Newton.jl")
include("normal/normal.jl")
include("Krylov_utils.jl")
include("ldl_dense.jl")

function init_pad!(pad::PreallocatedData)
  return pad
end
