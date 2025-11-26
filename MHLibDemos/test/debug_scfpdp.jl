using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using MHLibDemos

solve_scfpdp("ls",
    joinpath(@__DIR__, "..", "instances", "50", "train",
        "instance1_nreq50_nveh2_gamma50.txt"),
    neighborhood=:relocate, strategy=:best_improvement,)
