include("../src/SCFPDP.jl") 
include("../src/Genetic.jl")  

using MHLib
using Random
using StatsBase

filename = joinpath(@__DIR__, "..", "instances", "50", "test",
                    "instance32_nreq50_nveh2_gamma45.txt")
# filename = joinpath(@__DIR__, "..", "instances", "200", "test",
#                     "instance31_nreq200_nveh4_gamma192.txt")

# filename = joinpath(@__DIR__, "..", "instances", "1000", "test",
#                     "instance31_nreq1000_nveh20_gamma890.txt")

@info "Loading instance" filename
inst = SCFPDPInstance(filename)

sol = SCFPDPSolution(inst)

# run Genetic Algorithm
sol, stats = run_genetic!(
    sol;
    pop_size = 30,
    crossover_rate = 0.9,
    mutation_rate = 0.2,
    tournament_size = 3,
    elite_size = 2,
    ttime = 2.0,
    max_generations = 1000,
    construction_method = :nn_rand,
    seed = 1,
    verbose = true
)

println("\n=== Genetic Algorithm result ===")
println("Best objective: ", stats.best_val)
println("Generations:    ", stats.iters)
println("Runtime [s]:    ", stats.time)
println("Feasible:       ", is_feasible(sol))
println("Served count:   ", count(sol.served))  
