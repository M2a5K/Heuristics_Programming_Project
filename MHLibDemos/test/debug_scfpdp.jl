using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using MHLibDemos

solve_scfpdp("ls", seed=42)
