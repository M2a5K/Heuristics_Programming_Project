#!/usr/bin/env julia

using Pkg
Pkg.activate(".")

using MHLibDemos
using MHLib
using Glob
using CSV
using DataFrames
using Statistics

# ---------------------------
# SETTINGS
# ---------------------------
train_folder = joinpath(@__DIR__, "..", "..", "instances", "50", "train")
println("Tuning on: ", train_folder)

files = sort(glob("*.txt", train_folder))

# pick a small subset of training instances for tuning
tune_files = files[1:5]   # e.g. first 5

alphas = [0.1, 0.3, 0.5, 0.7]

# nn_rand_multi parameters
iters_list = [5, 10, 30]
rand_reps_tune = 5

# GRASP parameters
grasp_iters_list = [20, 50, 100]
grasp_reps_tune = 5

# ---------------------------
# Helper
# ---------------------------
function run_solver(alg, instance; kwargs...)
    t = @elapsed sol = solve_scfpdp(alg; filename = instance, kwargs...)
    return MHLib.obj(sol), t
end

# ---------------------------
# TUNING nn_rand_multi
# ---------------------------
rand_rows = DataFrame(
    instance  = String[],
    alpha     = Float64[],
    iters     = Int[],
    mean_obj  = Float64[],
    std_obj   = Float64[],
    mean_t    = Float64[],
)

for f in tune_files
    inst_name = split(basename(f), ".")[1]
    println("\n[Rand tuning] instance: ", inst_name)

    for α in alphas, iters in iters_list
        objs = Float64[]
        ts   = Float64[]

        for s in 1:rand_reps_tune
            obj, t = run_solver("nn_rand_multi", f;
                                alpha = α,
                                iters = iters,
                                seed  = s)
            push!(objs, obj)
            push!(ts, t)
        end

        push!(rand_rows, (
            inst_name,
            α,
            iters,
            mean(objs),
            std(objs),
            mean(ts),
        ))
    end
end

CSV.write(joinpath(@__DIR__, "tuning_rand_50_train.csv"), rand_rows)
println("Wrote tuning_rand_50_train.csv")

# ---------------------------
# TUNING GRASP
# ---------------------------
grasp_rows = DataFrame(
    instance  = String[],
    alpha     = Float64[],
    niters    = Int[],
    mean_obj  = Float64[],
    std_obj   = Float64[],
    mean_t    = Float64[],
)

for f in tune_files
    inst_name = split(basename(f), ".")[1]
    println("\n[GRASP tuning] instance: ", inst_name)

    for α in alphas, niters in grasp_iters_list
        objs = Float64[]
        ts   = Float64[]

        for s in 1:grasp_reps_tune
            obj, t = run_solver("grasp", f;
                                alpha  = α,
                                niters = niters,
                                seed   = s)
            push!(objs, obj)
            push!(ts, t)
        end

        push!(grasp_rows, (
            inst_name,
            α,
            niters,
            mean(objs),
            std(objs),
            mean(ts),
        ))
    end
end

CSV.write(joinpath(@__DIR__, "tuning_grasp_50_train.csv"), grasp_rows)
println("Wrote tuning_grasp_50_train.csv")
