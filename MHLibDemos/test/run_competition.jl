#!/usr/bin/env julia

using MHLibDemos
using MHLibDemos: SCFPDPInstance, decompose_objective
using MHLib
using Random
using DataFrames
using CSV

export run_competition


instances_competition = [
    # joinpath(@__DIR__, "..", "instances", "50", "competition",
    #     "instance61_nreq50_nveh2_gamma44.txt"),
    joinpath(@__DIR__, "..", "instances", "100", "competition",
        "instance61_nreq100_nveh2_gamma91.txt"),
    # joinpath(@__DIR__, "..", "instances", "200", "competition",
    #     "instance61_nreq200_nveh4_gamma191.txt"),
    # joinpath(@__DIR__, "..", "instances", "500", "competition",
    #     "instance61_nreq500_nveh10_gamma430.txt"),
    joinpath(@__DIR__, "..", "instances", "1000", "competition",
        "instance61_nreq1000_nveh20_gamma879.txt"),
    joinpath(@__DIR__, "..", "instances", "2000", "competition",
        "instance61_nreq2000_nveh40_gamma1829.txt"),
    # joinpath(@__DIR__, "..", "instances", "5000", "competition",
    #     "instance61_nreq5000_nveh100_gamma4448.txt"),
    # joinpath(@__DIR__, "..", "instances", "10000", "competition",
    #     "instance61_nreq10000_nveh200_gamma8803.txt"),
]

# Seeds to reduce variance
seeds  = 1:10

# Maximum number of iterations
titer = 100

# Local search parameters
# exclude delta evaluations for now, as it is not implemented for all neighborhoods
searchparams = [ [:two_opt, :inter_route, :relocate],
                    [:first_improvement, :best_improvement],
                    [false] ]

# Output CSV
out_csv = joinpath(@__DIR__, "results_competition.csv")

function run_for_instance(filename::AbstractString;
                                       seeds)
    # Read instance metadata (n, nk, C, gamma, rho, …)
    inst = SCFPDPInstance(filename)
    instname = splitext(basename(filename))[1]

    # Check if solution file exists and load it
    sol_files = glob("sol_$(instname)_nn_det*.txt", joinpath(@__DIR__, "results", "solutions"))
    if !isempty(sol_files)
        filename_sol = first(sol_files)
        init_sol = read_solution(filename_sol, inst, "competition")
    else
        init_sol = SCFPDPSolution(inst)
        construct_nn_det!(init_sol)
    end

    rows = NamedTuple[]

    for seed in seeds
        # Deterministic NN construction
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
                      alpha      = NaN,  
                      seed       = seed,
                      total_time = t_det,
                      fairness   = fair_det,
                      obj        = obj_det,
                      iterations = it_det,
                      runtime    = rt_det))

        # Randomized NN construction
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

        # GRASP (NN-rand + local search)
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
                        titer = 100,
                        lsparams = ls_params,
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
                sol, it, rt = solve_scfpdp("vnd", filename; seed, titer=100, lsparams=ls_params, initsol = init_sol)
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
                    iterations      = it,
                    runtime         = rt,
                ))
            end
        end
    end

    return rows
end

# -----------------------------------------------------
# Main: loop over all chosen instances
# -----------------------------------------------------

function run_competition()
    all_rows = NamedTuple[]

    for filename in instances_competition
        println("\n===============================")
        println("Running solver for competition on:")
        println("  $(filename)")
        println("===============================")

        if !isfile(filename)
            @warn "Instance file does not exist, skipping" filename
            continue
        end

        rows = run_for_instance(
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

# run_competition()
