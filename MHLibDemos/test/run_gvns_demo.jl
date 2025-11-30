using Pkg
# Pkg.activate("MHLibDemos")
Pkg.activate(".")


using MHLibDemos
using MHLibDemos: solve_scfpdp, decompose_objective

filename = joinpath("MHLibDemos", "instances", "50", "train",
                    "instance1_nreq50_nveh2_gamma50.txt")

sol_gen_vns, it_gen_vns, rt_gen_vns = solve_scfpdp(
    "gen_vns",
    filename;
    seed = 1,
    titer = 500,
    strategy = :first_improvement,
    use_delta = false,
)

println("GEN-VNS iterations: ", it_gen_vns)
println("GEN-VNS runtime:    ", rt_gen_vns, " s")

total_time, fairness, obj = decompose_objective(sol_gen_vns)
println("total_time = $total_time, fairness = $fairness, obj = $obj")
println(sol_gen_vns)
