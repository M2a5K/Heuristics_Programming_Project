#!/usr/bin/env julia

using MHLibDemos
using MHLibDemos: SCFPDPInstance, decompose_objective
using MHLib
using Random
using DataFrames
using CSV

export run11b

# Picking 6 instances (all from folder "test")
instances_test = [
    # size 50
    # joinpath(@__DIR__, "..", "instances", "50", "test",
    #          "instance31_nreq50_nveh2_gamma50.txt"),
    # joinpath(@__DIR__, "..", "instances", "50", "test",
    #          "instance36_nreq50_nveh2_gamma48.txt"),
    # joinpath(@__DIR__, "..", "instances", "50", "test",
    #          "instance41_nreq50_nveh2_gamma45.txt"),
    # joinpath(@__DIR__, "..", "instances", "50", "test",
    #          "instance46_nreq50_nveh2_gamma47.txt"),
    # joinpath(@__DIR__, "..", "instances", "50", "test",
    #          "instance51_nreq50_nveh2_gamma45.txt"),
    # joinpath(@__DIR__, "..", "instances", "50", "test",
    #          "instance56_nreq50_nveh2_gamma43.txt"),

    # size 200
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

    # size 1000
    joinpath(@__DIR__, "..", "instances", "1000", "test",
             "instance31_nreq1000_nveh20_gamma890.txt"),
    joinpath(@__DIR__, "..", "instances", "1000", "test",
             "instance36_nreq1000_nveh20_gamma909.txt"),
    joinpath(@__DIR__, "..", "instances", "1000", "test",
             "instance41_nreq1000_nveh20_gamma861.txt"),
    joinpath(@__DIR__, "..", "instances", "1000", "test",
             "instance46_nreq1000_nveh20_gamma904.txt"),
    joinpath(@__DIR__, "..", "instances", "1000", "test",
             "instance51_nreq1000_nveh20_gamma929.txt"),
    joinpath(@__DIR__, "..", "instances", "1000", "test",
             "instance56_nreq1000_nveh20_gamma867.txt"),

    # size 5000
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

# Local search parameters
# exclude delta evaluations for now, as it is not implemented for all neighborhoods
searchparams = [ [:two_opt, :inter_route, :relocate],
                    [:first_improvement, :best_improvement],
                    [false] ]

# Output CSV
out_csv = joinpath(@__DIR__, "results_11b.csv")

function run_search_for_instance(filename::AbstractString;
                                       seeds)
    # Read instance metadata (n, nk, C, gamma, rho, …)
    inst = SCFPDPInstance(filename)
    instname = splitext(basename(filename))[1]

    # Check if solution file exists and load it
    sol_files = glob("sol_$(instname)_nn_det*.txt", joinpath(@__DIR__, "results", "solutions"))
    if !isempty(sol_files)
        filename_sol = first(sol_files)
        init_sol = read_solution(filename_sol, inst, "test")
    else
        init_sol = SCFPDPSolution(inst)
        construct_nn_det!(init_sol)
    end

    rows = NamedTuple[]

    for seed in seeds
        # -------------------------------
        # Local Search with different neighborhoods and strategies
        # -------------------------------
        for neighborhood in searchparams[1]
            for strategy in searchparams[2]
                for use_delta in searchparams[3]
                    ls_params = MHLibDemos.LocalSearchParams(neighborhood, strategy, use_delta)
                    sol, it, rt = MHLibDemos.solve_scfpdp(
                        "ls",
                        filename;
                        seed  = seed,
                        lsparams = ls_params,
                        titer = 100,
                        initsol = init_sol,
                    )

                    t, fair, obj = decompose_objective(sol)

                    push!(rows, (; 
                        instance        = instname,
                        filename        = filename,
                        n               = inst.n,
                        nk              = inst.nk,
                        gamma           = inst.gamma,
                        C               = inst.C,
                        rho             = inst.rho,
                        alg             = "ls",
                        seed            = seed,
                        neighborhood    = neighborhood,
                        strategy        = strategy,
                        use_delta       = use_delta,
                        total_time      = t,
                        fairness        = fair,
                        obj             = obj,
                        iterations      = it,
                        runtime         = rt,
                    ))
                end
            end
        end


        # -------------------------------
        # VND with different strategies
        # -------------------------------
        for strategy in searchparams[2]
            for use_delta in searchparams[3]
                # the neighborhood does not matter here, because VND uses all of them;
                # still, we need to provide something to make the LocalSearchParams struct happy
                ls_params = MHLibDemos.LocalSearchParams(:two_opt, strategy, use_delta)
                sol, iters, time = solve_scfpdp("vnd", filename; seed, titer=100, lsparams=ls_params, initsol = init_sol)
                t, fair, obj = decompose_objective(sol)
                inst = sol.inst  # same instance, but nice and explicit

                push!(rows, (; 
                    instance        = instname,
                    filename        = filename,
                    n               = inst.n,
                    nk              = inst.nk,
                    gamma           = inst.gamma,
                    C               = inst.C,
                    rho             = inst.rho,
                    alg             = "vnd",
                    seed            = seed,
                    neighborhood    = :all,
                    strategy        = strategy,
                    use_delta       = use_delta,
                    total_time      = t,
                    fairness        = fair,
                    obj             = obj,
                    iterations      = iters,
                    runtime         = time,
                ))
            end
        end
    end

    return rows
end

# -----------------------------------------------------
# Main: loop over all chosen instances
# -----------------------------------------------------

function run11b()
    all_rows = NamedTuple[]

    for filename in instances_test
        println("\n===============================")
        println("Running search-based experiments on:")
        println("  $(filename)")
        println("===============================")

        if !isfile(filename)
            @warn "Instance file does not exist, skipping" filename
            continue
        end

        rows = run_search_for_instance(
            filename;
            seeds = seeds,
        )

        append!(all_rows, rows)
    end

    df = DataFrame(all_rows)
    println("\nWriting results to: $out_csv")
    CSV.write(out_csv, df)

    println("\nDone. Summary:")
    println(df)
end

# run11b()
