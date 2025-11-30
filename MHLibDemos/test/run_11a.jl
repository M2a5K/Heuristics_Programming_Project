#!/usr/bin/env julia

using MHLibDemos
using MHLibDemos: SCFPDPInstance, decompose_objective
using MHLib
using Random
using DataFrames
using CSV

# Picking 6 instances (all from folder "test")
instances = [
    # size 50
    joinpath(@__DIR__, "..", "instances", "50", "test",
             "instance31_nreq50_nveh2_gamma50.txt"),
    joinpath(@__DIR__, "..", "instances", "50", "test",
             "instance36_nreq50_nveh2_gamma48.txt"),
    joinpath(@__DIR__, "..", "instances", "50", "test",
             "instance41_nreq50_nveh2_gamma45.txt"),
    joinpath(@__DIR__, "..", "instances", "50", "test",
             "instance46_nreq50_nveh2_gamma47.txt"),
    joinpath(@__DIR__, "..", "instances", "50", "test",
             "instance51_nreq50_nveh2_gamma45.txt"),
    joinpath(@__DIR__, "..", "instances", "50", "test",
             "instance56_nreq50_nveh2_gamma43.txt"),

    # # size 200
    # joinpath(@__DIR__, "..", "instances", "200", "test",
    #          "instance31_nreq200_nveh4_gamma192.txt"),
    # joinpath(@__DIR__, "..", "instances", "200", "test",
    #          "instance36_nreq200_nveh4_gamma181.txt"),
    # joinpath(@__DIR__, "..", "instances", "200", "test",
    #          "instance41_nreq200_nveh4_gamma196.txt"),
    # joinpath(@__DIR__, "..", "instances", "200", "test",
    #          "instance46_nreq200_nveh4_gamma178.txt"),
    # joinpath(@__DIR__, "..", "instances", "200", "test",
    #          "instance51_nreq200_nveh4_gamma172.txt"),
    # joinpath(@__DIR__, "..", "instances", "200", "test",
    #          "instance56_nreq200_nveh4_gamma171.txt"),

    # # size 1000
    # joinpath(@__DIR__, "..", "instances", "1000", "test",
    #          "instance31_nreq1000_nveh20_gamma890.txt"),
    # joinpath(@__DIR__, "..", "instances", "1000", "test",
    #          "instance36_nreq1000_nveh20_gamma909.txt"),
    # joinpath(@__DIR__, "..", "instances", "1000", "test",
    #          "instance41_nreq1000_nveh20_gamma861.txt"),
    # joinpath(@__DIR__, "..", "instances", "1000", "test",
    #          "instance46_nreq1000_nveh20_gamma904.txt"),
    # joinpath(@__DIR__, "..", "instances", "1000", "test",
    #          "instance51_nreq1000_nveh20_gamma929.txt"),
    # joinpath(@__DIR__, "..", "instances", "1000", "test",
    #          "instance56_nreq1000_nveh20_gamma867.txt"),

    # # size 5000
    # joinpath(@__DIR__, "..", "instances", "5000", "test",
    #          "instance31_nreq5000_nveh100_gamma4468.txt"),
    # joinpath(@__DIR__, "..", "instances", "5000", "test",
    #          "instance36_nreq5000_nveh100_gamma4547.txt"),
    # joinpath(@__DIR__, "..", "instances", "5000", "test",
    #          "instance41_nreq5000_nveh100_gamma4652.txt"),
    # joinpath(@__DIR__, "..", "instances", "5000", "test",
    #          "instance46_nreq5000_nveh100_gamma4481.txt"),
    # joinpath(@__DIR__, "..", "instances", "5000", "test",
    #          "instance51_nreq5000_nveh100_gamma4485.txt"),
    # joinpath(@__DIR__, "..", "instances", "5000", "test",
    #          "instance56_nreq5000_nveh100_gamma4318.txt"),
]

# Seeds to reduce variance
seeds  = 1:10

# α-values for randomized construction / GRASP
alphas = [0.3]

# Number of GVNS iterations for the construction heuristics
titer_construction = 50

# Number of GRASP outer iterations (how many constructs + LS runs)
niters_grasp = 50

# Output CSV
out_csv = joinpath(@__DIR__, "results_11a.csv")

function run_construction_for_instance(filename::AbstractString;
                                       seeds,
                                       alphas,
                                       titer::Int)
    # Read instance metadata (n, nk, C, gamma, rho, …)
    inst = SCFPDPInstance(filename)
    instname = splitext(basename(filename))[1]

    rows = NamedTuple[]

    for seed in seeds
        # -------------------------------
        # Deterministic NN construction
        # -------------------------------
        sol_det, it_det, rt_det = MHLibDemos.solve_scfpdp(
            "nn_det",
            filename;
            seed  = seed,
            titer = titer,
        )

        t_det, fair_det, obj_det = decompose_objective(sol_det)

        push!(rows, (; instance   = instname,
                      filename   = filename,
                      n          = inst.n,
                      nK         = inst.nk,
                      gamma      = inst.gamma,
                      C          = inst.C,
                      rho        = inst.rho,
                      alg        = "nn_det",
                      alpha      = NaN,         # not used
                      seed       = seed,
                      total_time = t_det,
                      fairness   = fair_det,
                      obj        = obj_det,
                      iterations = it_det,
                      runtime    = rt_det))

        # -------------------------------
        # Randomized NN construction
        # -------------------------------
        for α in alphas
            sol_rand, it_rand, rt_rand = MHLibDemos.solve_scfpdp(
                "nn_rand",
                filename;
                seed  = seed,
                titer = titer,
                alpha = α,
            )

            t_r, f_r, o_r = decompose_objective(sol_rand)

            push!(rows, (; instance   = instname,
                          filename   = filename,
                          n          = inst.n,
                          nK         = inst.nk,
                          gamma      = inst.gamma,
                          C          = inst.C,
                          rho        = inst.rho,
                          alg        = "nn_rand",
                          alpha      = α,
                          seed       = seed,
                          total_time = t_r,
                          fairness   = f_r,
                          obj        = o_r,
                          iterations = it_rand,
                          runtime    = rt_rand))
        end

        # -------------------------------
        # GRASP (NN-rand + local search)
        # -------------------------------
        for α in alphas
            sol_grasp, it_grasp, rt_grasp = MHLibDemos.solve_scfpdp(
                "grasp",
                filename;
                seed        = seed,
                titer       = titer,
                niters      = niters_grasp,
                alpha       = α,
                neighborhood = :two_opt,
                strategy     = :first_improvement,
                use_delta    = false,
            )

            t_g, f_g, o_g = decompose_objective(sol_grasp)

            push!(rows, (; instance   = instname,
                          filename   = filename,
                          n          = inst.n,
                          nK         = inst.nk,
                          gamma      = inst.gamma,
                          C          = inst.C,
                          rho        = inst.rho,
                          alg        = "grasp",
                          alpha      = α,
                          seed       = seed,
                          total_time = t_g,
                          fairness   = f_g,
                          obj        = o_g,
                          iterations = it_grasp,
                          runtime    = rt_grasp))
        end
    end

    return rows
end

# -----------------------------------------------------
# Main: loop over all chosen instances
# -----------------------------------------------------

function main()
    all_rows = NamedTuple[]

    for filename in instances
        println("\n===============================")
        println("Running construction experiments on:")
        println("  $(filename)")
        println("===============================")

        if !isfile(filename)
            @warn "Instance file does not exist, skipping" filename
            continue
        end

        rows = run_construction_for_instance(
            filename;
            seeds  = seeds,
            alphas = alphas,
            titer  = titer_construction,
        )

        append!(all_rows, rows)
    end

    df = DataFrame(all_rows)
    println("\nWriting results to: $out_csv")
    CSV.write(out_csv, df)

    println("\nDone. Summary:")
    println(df)
end

main()
