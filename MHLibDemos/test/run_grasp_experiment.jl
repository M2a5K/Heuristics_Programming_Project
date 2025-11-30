#!/usr/bin/env julia

using MHLibDemos
using MHLibDemos: decompose_objective
using Random
using DataFrames
using CSV

function tune_grasp(;
        root50  = joinpath(@__DIR__, "..", "instances", "50", "train"),
        root200 = joinpath(@__DIR__, "..", "instances", "200", "train"),

        inst50  = ["instance1_nreq50_nveh2_gamma50.txt",
                   "instance3_nreq50_nveh2_gamma49.txt",
                   "instance7_nreq50_nveh2_gamma41.txt",
                   "instance14_nreq50_nveh2_gamma45.txt",
                   "instance27_nreq50_nveh2_gamma41.txt"],
        inst200 = ["instance1_nreq200_nveh4_gamma177.txt",
                   "instance3_nreq200_nveh4_gamma180.txt",
                   "instance7_nreq200_nveh4_gamma175.txt",
                   "instance10_nreq200_nveh4_gamma182.txt",
                   "instance14_nreq200_nveh4_gamma181.txt"],

        alpha   = 0.3,
        niters_list = [10,20,50],
        seeds   = 1:3,
    )

    rows = NamedTuple[]

    for (root, inst_list) in ((root50, inst50), (root200, inst200))
        for inst_file in inst_list
            filename = joinpath(root, inst_file)

            for niters in niters_list, seed in seeds
                sol, iters, time = solve_scfpdp("grasp", filename;
                                                seed   = seed,
                                                alpha  = alpha,
                                                niters = niters)

                t, fair, obj = decompose_objective(sol)
                push!(rows, (; 
                    instance   = inst_file,
                    alpha      = alpha,
                    niters     = niters,
                    seed       = seed,
                    obj        = obj,
                    total_time = t,
                    fairness   = fair,
                    runtime    = time,
                ))
            end
        end
    end

    df = DataFrame(rows)
    outdir = joinpath(@__DIR__, "results")
    mkpath(outdir)
    outfile = joinpath(outdir, "grasp_tuning.csv")
    CSV.write(outfile, df)
    println("Saved: $outfile")
    println(first(df, 10))
end

# tune_grasp()