#!/usr/bin/env julia

using Pkg
Pkg.activate(".")

using MHLibDemos
using MHLib
using Glob
using CSV
using DataFrames
using Statistics

# -----------------------------------------------------
# SETTINGS
# -----------------------------------------------------

# folder with instances (relative to this script)
folder = joinpath(@__DIR__, "..", "instances", "50", "test")
println("Looking for files in: ", folder)

# where to write results
output_file = joinpath(@__DIR__, "results_50.csv")

rand_k      = 10      # nn_rand_multi: iterations per run
rand_reps   = 5       # how many independent runs for nn_rand_multi
grasp_iters = 50      # GRASP iterations
alpha       = 0.3     # RCL parameter for randomized constructions

# -----------------------------------------------------
# Helper to safely run timing + obj
# -----------------------------------------------------
function run_solver(alg, instance; kwargs...)
    t = @elapsed sol = solve_scfpdp(alg; filename = instance, kwargs...)
    return MHLib.obj(sol), t
end

# -----------------------------------------------------
# Collect all instance files
# -----------------------------------------------------
files = sort(glob("*.txt", folder))
println("Found ", length(files), " instance files.")
if isempty(files)
    error("No instance files found in $folder")
end

results = DataFrame(
    instance  = String[],
    det_obj   = Float64[],
    det_t     = Float64[],
    rand_obj  = Float64[],
    rand_std  = Float64[],
    rand_t    = Float64[],
    grasp_obj = Float64[],
    grasp_t   = Float64[],
)

# -----------------------------------------------------
# Main loop
# -----------------------------------------------------
for file in files
    inst_name = split(basename(file), ".")[1]
    println("\n==============================")
    println("Running on: $inst_name")
    println("==============================")

    # --- Deterministic greedy (nn_det) ---
    det_obj, det_t = run_solver("nn_det", file)

    # --- Random greedy multi-start (nn_rand_multi) ---
    objs = Float64[]
    ts   = Float64[]

    for s in 1:rand_reps
        obj, t = run_solver("nn_rand_multi", file;
                            iters = rand_k,
                            alpha = alpha,
                            seed  = s)
        push!(objs, obj)
        push!(ts,   t)
    end

    rand_obj = mean(objs)
    rand_std = std(objs)
    rand_t   = mean(ts)

    # --- GRASP ---
    grasp_obj, grasp_t = run_solver("grasp", file;
                                    niters = grasp_iters,
                                    alpha  = alpha,
                                    seed   = 1234)

    push!(results, (
        inst_name,
        det_obj,
        det_t,
        rand_obj,
        rand_std,
        rand_t,
        grasp_obj,
        grasp_t,
    ))
end

# -----------------------------------------------------
# Save to CSV
# -----------------------------------------------------
CSV.write(output_file, results)
println("\nSaved results to $output_file")
