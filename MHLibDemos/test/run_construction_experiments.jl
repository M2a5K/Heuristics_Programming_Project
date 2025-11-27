# run_construction.jl
#
# Run construction-heuristic experiments for several instances and α-values.
#
# IMPORTANT: start Julia with one thread, e.g.
#   JULIA_NUM_THREADS=1 julia

using MHLibDemos
using MHLibDemos: decompose_objective
using Random
using DataFrames
using CSV

function run_construction_experiments(; 
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

    rows = NamedTuple[]

    for inst_file in inst_files
        filename = joinpath(rootdir, inst_file)

        for seed in seeds
            # --- deterministic NN construction ---
            sol_det, iters, time = solve_scfpdp("nn_det", filename; seed)
            t_det, fair_det, obj_det = decompose_objective(sol_det)
            inst = sol_det.inst

            push!(rows, (; 
                instance  = inst_file,
                n         = inst.n,
                nk        = inst.nk,
                gamma     = inst.gamma,
                C         = inst.C,
                rho       = inst.rho,
                alg       = "nn_det",
                alpha     = NaN,
                seed      = seed,
                total_time = t_det,
                fairness   = fair_det,
                obj        = obj_det,
                iters      = iters,
                time       = time,
                # TODO rename this to 'runtime'? (to avoid confusion with total_time)
            ))

            # --- randomized NN construction for each α ---
            for α in alphas
                sol_rand, iters, time = solve_scfpdp("nn_rand", filename; seed, alpha=α)
                t_r, f_r, o_r = decompose_objective(sol_rand)
                inst = sol_rand.inst  # same instance, but nice and explicit

                push!(rows, (; 
                    instance  = inst_file,
                    n         = inst.n,
                    nk        = inst.nk,
                    gamma     = inst.gamma,
                    C         = inst.C,
                    rho       = inst.rho,
                    alg       = "nn_rand",
                    alpha     = α,
                    seed      = seed,
                    total_time = t_r,
                    fairness   = f_r,
                    obj        = o_r,
                    iters      = iters,
                    time       = time,
                    # TODO rename this to 'runtime'? (to avoid confusion with total_time)
                ))
            end
        end
    end

    df = DataFrame(rows)

    outdir = joinpath(@__DIR__, "results")
    mkpath(outdir)

    outfile = joinpath(outdir, "construction_results_experiments.csv")
    CSV.write(outfile, df)

    println("Results saved to: $outfile")
    println(first(df, 10))  # show a small preview
end

# run_construction_experiments()
