import Pkg

# Activate project in repo root
Pkg.activate(@__DIR__)

# Install Julia dependencies needed for the benchmarks
Pkg.add([
    "ChenSignatures",
    "BenchmarkTools",
    "StaticArrays",
    "YAML",
    "Revise",
])

Pkg.precompile()
