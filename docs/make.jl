using Documenter, RipQP

makedocs(
  modules = [RipQP],
  doctest = true,
  linkcheck = true,
  strict = true,
  format = Documenter.HTML(
    assets = ["assets/style.css"],
    prettyurls = get(ENV, "CI", nothing) == "true",
  ),
  sitename = "RipQP.jl",
  pages = ["Home" => "index.md", "API" => "API.md", "Tutorial" => "tutorial.md", "Reference" => "reference.md"],
)

deploydocs(
  deps = nothing,
  make = nothing,
  repo = "github.com/JuliaSmoothOptimizers/RipQP.jl.git",
  target = "build",
  devbranch = "master",
)
