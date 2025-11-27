# Run search-heuristic experiments for several instances and α-values.
#
# IMPORTANT: start Julia with one thread, e.g.
#   JULIA_NUM_THREADS=1 julia

using MHLibDemos
using MHLibDemos: decompose_objective
using Random
using DataFrames
using CSV

function run_search_experiments(; 
        rootdir  = joinpath(@__DIR__, "..", "instances", "50", "train"),
        inst_files = [
            "instance1_nreq50_nveh2_gamma50.txt",
            "instance3_nreq50_nveh2_gamma49.txt",
            "instance7_nreq50_nveh2_gamma41.txt",
            "instance14_nreq50_nveh2_gamma45.txt",
            "instance27_nreq50_nveh2_gamma41.txt",
        ],
        seeds   = 1:10,
        alphas  = [0.0, 0.3, 0.7])

    searchparams = [ [:two_opt, :inter_route, :relocate],
                     [:first_improvement, :best_improvement],
                     [false, true] ]

    rows = NamedTuple[]

    for inst_file in inst_files
        filename = joinpath(rootdir, inst_file)

        for seed in seeds
            # --- local search ---
            for neighborhood in searchparams[1]
                for strategy in searchparams[2]
                    for use_delta in searchparams[3]
                        ls_params = MHLibDemos.LocalSearchParams(neighborhood, strategy, use_delta)
                        sol, iters, time = solve_scfpdp("ls", filename; seed, ls_params)
                        t, fair, obj = decompose_objective(sol)
                        inst = sol.inst

                        push!(rows, (; 
                            instance        = inst_file,
                            n               = inst.n,
                            nk              = inst.nk,
                            gamma           = inst.gamma,
                            C               = inst.C,
                            rho             = inst.rho,
                            alg             = "ls",
                            alpha           = NaN,
                            seed            = seed,
                            neighborhood    = neighborhood,
                            strategy        = strategy,
                            use_delta       = use_delta,
                            total_time      = t,
                            fairness        = fair,
                            obj             = obj,
                            iters           = iters,
                            runtime         = time,
                        ))
                    end
                end
            end

            # --- variable neighborhood descent (VND) ---
            
            # TODO to be implemented...

            # for α in alphas
            #     sol, iters, time = solve_scfpdp("vnd", filename; seed, alpha=α)
            #     t_r, f_r, o_r = decompose_objective(sol_rand)
            #     inst = sol_rand.inst  # same instance, but nice and explicit

            #     push!(rows, (; 
            #         instance  = inst_file,
            #         n         = inst.n,
            #         nk        = inst.nk,
            #         gamma     = inst.gamma,
            #         C         = inst.C,
            #         rho       = inst.rho,
            #         alg       = "nn_rand",
            #         alpha     = α,
            #         seed      = seed,
            #         total_time = t_r,
            #         fairness   = f_r,
            #         obj        = o_r,
            #         iters      = iters,
            #         time       = time,
            #     ))
            # end
        end
    end

    df = DataFrame(rows)

    outdir = joinpath(@__DIR__, "results")
    mkpath(outdir)

    outfile = joinpath(outdir, "search_results_experiments.csv")
    CSV.write(outfile, df)

    println("Results saved to: $outfile")
    println(first(df, 10))  # show a small preview
end

# run_search_experiments()
