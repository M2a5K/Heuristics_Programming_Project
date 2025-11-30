# run_construction_experiments.jl

using MHLibDemos
using MHLibDemos: decompose_objective
using Random
using DataFrames
using CSV

function run_construction_experiments(; 
    # ORIGINAL folder
    root50  = joinpath(@__DIR__, "..", "instances", "50", "train"),
    # NEW folder
    root200 = joinpath(@__DIR__, "..", "instances", "200", "train"),

    # ORIGINAL 5 INSTANCES
    inst50 = [
        "instance1_nreq50_nveh2_gamma50.txt",
        "instance3_nreq50_nveh2_gamma49.txt",
        "instance7_nreq50_nveh2_gamma41.txt",
        "instance14_nreq50_nveh2_gamma45.txt",
        "instance27_nreq50_nveh2_gamma41.txt",
    ],

    # NEW 5 INSTANCES FROM 200/train (fixed, not random)
    inst200 = [
        "instance1_nreq200_nveh4_gamma177.txt",
        "instance3_nreq200_nveh4_gamma180.txt",
        "instance7_nreq200_nveh4_gamma175.txt",
        "instance10_nreq200_nveh4_gamma182.txt",
        "instance14_nreq200_nveh4_gamma181.txt",
    ],

    seeds   = 1:10,
    alphas  = [0.0, 0.3, 0.7]
)

    rows = NamedTuple[]

    # Combine everything
    all_instances = vcat(
        [(root50, f) for f in inst50]...,
        [(root200, f) for f in inst200]...
    )

    for (dirpath, inst_file) in all_instances
        filename = joinpath(dirpath, inst_file)

        for seed in seeds

            # ---- deterministic ----
            sol_det, iters, time = solve_scfpdp("nn_det", filename; seed)
            t_det, fair_det, obj_det = decompose_objective(sol_det)
            inst = sol_det.inst

            push!(rows, (; 
                instance   = inst_file,
                n          = inst.n,
                nk         = inst.nk,
                gamma      = inst.gamma,
                C          = inst.C,
                rho        = inst.rho,
                alg        = "nn_det",
                alpha      = NaN,
                seed       = seed,
                total_time = t_det,
                fairness   = fair_det,
                obj        = obj_det,
                iters      = iters,
                runtime    = time,
            ))

            # ---- randomized for α ----
            for α in alphas
                sol_rand, iters, time = solve_scfpdp("nn_rand", filename; seed, alpha=α)
                t_r, f_r, o_r = decompose_objective(sol_rand)
                inst = sol_rand.inst

                push!(rows, (; 
                    instance   = inst_file,
                    n          = inst.n,
                    nk         = inst.nk,
                    gamma      = inst.gamma,
                    C          = inst.C,
                    rho        = inst.rho,
                    alg        = "nn_rand",
                    alpha      = α,
                    seed       = seed,
                    total_time = t_r,
                    fairness   = f_r,
                    obj        = o_r,
                    iters      = iters,
                    runtime    = time,
                ))
            end
        end
    end

    df = DataFrame(rows)

    outdir = joinpath(@__DIR__, "results")
    mkpath(outdir)

    outfile = joinpath(outdir, "construction_results_experiments1.csv")
    CSV.write(outfile, df)

    println("Results saved to: $outfile")
    println(first(df, 10))
end

# run_construction_experiments()
