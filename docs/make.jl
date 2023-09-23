using Documenter, RipQP

makedocs(
  modules = [RipQP],
  doctest = true,
  linkcheck = true,
  format = Documenter.HTML(
    assets = ["assets/style.css"],
    prettyurls = get(ENV, "CI", nothing) == "true",
  ),
  sitename = "RipQP.jl",
  pages = [
    "Home" => "index.md",
    "API" => "API.md",
    "Tutorial" => "tutorial.md",
    "Switching solvers" => "switch_solv.md",
    "Multi-precision" => "multi_precision.md",
    "Reference" => "reference.md",
  ],
)

deploydocs(
  deps = nothing,
  make = nothing,
  repo = "github.com/JuliaSmoothOptimizers/RipQP.jl.git",
  target = "build",
  devbranch = "main",
  push_preview = true,
)
