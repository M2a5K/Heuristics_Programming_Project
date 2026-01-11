include("../src/SCFPDP.jl") 
include("../src/ACO.jl")  

using MHLib
using Random
using StatsBase # this needs to be changed to our gvns framekwo 

filename = joinpath(@__DIR__, "..", "instances", "50", "test",
                    "instance31_nreq50_nveh2_gamma50.txt")
# filename = joinpath(@__DIR__, "..", "instances", "200", "test",
#                     "instance31_nreq200_nveh4_gamma192.txt")

# filename = joinpath(@__DIR__, "..", "instances", "1000", "test",
#                     "instance31_nreq1000_nveh20_gamma890.txt")

@info "Loading instance" filename
inst = SCFPDPInstance(filename)

sol = SCFPDPSolution(inst)

# run ACO
sol, stats = run_aco!(
    sol;
    num_ants = 5,
    alpha = 1.0,
    beta = 3.0,
    rho = 0.1,
    ttime = 2.0,
    seed = 1,
    tau0 = 1.0,
    Q = 0.2, # modify if needed for deposit_pheromones_best!
    # aco_lsparams=MHLibDemos.LocalSearchParams(:two_opt, :first_improvement, false),
    # aco_ls_iters=2,
)

println("\n=== ACO result ===")
println("Best objective: ", stats.best_val)
println("Iterations:     ", stats.iters)
println("Runtime [s]:    ", stats.time)
println("Feasible:       ", is_feasible(sol))
println("Served count:   ", count(sol.served))  
