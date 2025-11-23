JULIA_NUM_THREADS=1


using MHLibDemos
using MHLibDemos: decompose_objective
using Random
using DataFrames
using CSV

function run_construction_experiments(; 
        filename = joinpath(@__DIR__, "..", "instances", "50", "train",
                            "instance1_nreq50_nveh2_gamma50.txt"),
        seeds   = 1:10,
        alphas  = [0.0, 0.3, 0.7])

    rows = NamedTuple[]

    for seed in seeds
        sol_det = solve_scf_pdp("nn_det"; filename, seed)
        t_det, fair_det, obj_det = decompose_objective(sol_det)
        push!(rows, (; alg="nn_det", alpha=NaN, seed, total_time=t_det,
                     fairness=fair_det, obj=obj_det))

        for α in alphas
            sol_rand = solve_scf_pdp("nn_rand"; filename, seed, alpha=α)
            t_r, f_r, o_r = decompose_objective(sol_rand)
            push!(rows, (; alg="nn_rand", alpha=α, seed, total_time=t_r,
                         fairness=f_r, obj=o_r))
        end

        sol_pilot = solve_scf_pdp("pilot"; filename, seed)
        t_p, f_p, o_p = decompose_objective(sol_pilot)
        push!(rows, (; alg="pilot", alpha=NaN, seed, total_time=t_p,
                     fairness=f_p, obj=o_p))
    end

    df = DataFrame(rows)
    # CSV.write("construction_results.csv", df)
    outdir = joinpath(@__DIR__, "results")
    mkpath(outdir)

    outfile = joinpath(outdir, "construction_results.csv")
    CSV.write(outfile, df)

    println("Results saved to: $outfile")
    # println(df)
end

run_construction_experiments()
