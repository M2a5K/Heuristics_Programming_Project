#!/usr/bin/env julia
using MHLibDemos
using MHLibDemos: SCFPDPInstance, decompose_objective
using DataFrames, CSV
using Statistics

const SEEDS = 1:10
const TITER = 20

const INSTANCES_TEST_50 = [
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

]

const INSTANCES_TEST_100 = [
    joinpath(@__DIR__, "..", "instances", "100", "test",
             "instance31_nreq100_nveh2_gamma86.txt"),
    joinpath(@__DIR__, "..", "instances", "100", "test",
             "instance36_nreq100_nveh2_gamma90.txt"),
    joinpath(@__DIR__, "..", "instances", "100", "test",
             "instance41_nreq100_nveh2_gamma86.txt"),
    joinpath(@__DIR__, "..", "instances", "100", "test",
             "instance46_nreq100_nveh2_gamma92.txt"),
    joinpath(@__DIR__, "..", "instances", "100", "test",
             "instance51_nreq100_nveh2_gamma93.txt"),
    joinpath(@__DIR__, "..", "instances", "100", "test",
             "instance56_nreq100_nveh2_gamma89.txt"),

]

const INSTANCES_TEST_200 = [
    joinpath(@__DIR__, "..", "instances", "200", "test",
             "instance31_nreq200_nveh4_gamma192.txt"),
    joinpath(@__DIR__, "..", "instances", "200", "test",
             "instance36_nreq200_nveh4_gamma181.txt"),
    joinpath(@__DIR__, "..", "instances", "200", "test",
             "instance41_nreq200_nveh4_gamma196.txt"),
    joinpath(@__DIR__, "..", "instances", "200", "test",
             "instance46_nreq200_nveh4_gamma178.txt"),
    joinpath(@__DIR__, "..", "instances", "200", "test",
             "instance51_nreq200_nveh4_gamma172.txt"),
    joinpath(@__DIR__, "..", "instances", "200", "test",
             "instance56_nreq200_nveh4_gamma171.txt"),

]

const INSTANCES_TEST_1000 = [
    joinpath(@__DIR__, "..", "instances", "1000", "test", "instance31_nreq1000_nveh20_gamma890.txt"),
    joinpath(@__DIR__, "..", "instances", "1000", "test", "instance36_nreq1000_nveh20_gamma909.txt"),
    joinpath(@__DIR__, "..", "instances", "1000", "test", "instance41_nreq1000_nveh20_gamma861.txt"),
    joinpath(@__DIR__, "..", "instances", "1000", "test", "instance46_nreq1000_nveh20_gamma904.txt"),
    joinpath(@__DIR__, "..", "instances", "1000", "test", "instance51_nreq1000_nveh20_gamma929.txt"),
    joinpath(@__DIR__, "..", "instances", "1000", "test", "instance56_nreq1000_nveh20_gamma867.txt"),
]


# Genetic Algorithm tuned params for size 50 and 1000 (irace rank 1 config)
const POP_SIZE          = 28
const CROSSOVER_RATE    = 0.309723
const MUTATION_RATE     = 0.618255
const TOURNAMENT_SIZE   = 7
const ELITE_SIZE        = 5

# # Genetic Algorithm tuned params for size 100 (irace rank 1 config)
# const POP_SIZE          = 30
# const CROSSOVER_RATE    = 0.149789
# const MUTATION_RATE     = 0.851321
# const TOURNAMENT_SIZE   = 9
# const ELITE_SIZE        = 4

# # Genetic Algorithm tuned params for size 200 (irace rank 1 config)
# const POP_SIZE          = 49
# const CROSSOVER_RATE    = 0.105829
# const MUTATION_RATE     = 0.131549
# const TOURNAMENT_SIZE   = 5
# const ELITE_SIZE        = 4


const OUT_CSV_RAW = joinpath(@__DIR__, "results_test_genetic_raw_1000.csv")
const OUT_CSV_SUM = joinpath(@__DIR__, "results_test_genetic_summary_1000.csv")


function run_genetic_test(instances::Vector{String})
    rows = NamedTuple[]

    for filename in instances
        instname = splitext(basename(filename))[1]
        if !isfile(filename)
            @warn "Missing instance file, skipping" filename
            continue
        end

        inst = SCFPDPInstance(filename)

        println("\n[GenAlg] Running: $instname")
        for seed in SEEDS
            sol, it, rt = MHLibDemos.solve_scfpdp(
                "genetic",
                filename;
                seed            = seed,
                titer           = TITER,
                pop_size        = POP_SIZE,
                crossover_rate  = CROSSOVER_RATE,
                mutation_rate   = MUTATION_RATE,
                tournament_size = TOURNAMENT_SIZE,
                elite_size      = ELITE_SIZE,
            )

            total_time, fairness, obj = decompose_objective(sol)

            push!(rows, (;
                instance        = instname,
                filename        = filename,
                n               = inst.n,
                nK              = inst.nk,
                gamma           = inst.gamma,
                C               = inst.C,
                rho             = inst.rho,
                alg             = "genetic",
                seed            = seed,
                pop_size        = POP_SIZE,
                crossover_rate  = CROSSOVER_RATE,
                mutation_rate   = MUTATION_RATE,
                tournament_size = TOURNAMENT_SIZE,
                elite_size      = ELITE_SIZE,
                total_time      = total_time,
                fairness        = fairness,
                obj             = obj,
                iterations      = it,
                runtime         = rt
            ))
        end
    end

    df = DataFrame(rows)
    CSV.write(OUT_CSV_RAW, df)

    g = groupby(df, [:instance, :n])
    df_sum = combine(g,
        :obj => mean => :obj_mean,
        :obj => std  => :obj_std,
        :obj => minimum => :obj_min,
        :runtime => mean => :runtime_mean,
        :runtime => maximum => :runtime_max,
        :fairness => mean => :fairness_mean
    )
    CSV.write(OUT_CSV_SUM, df_sum)

    println("\nWrote raw:  $OUT_CSV_RAW")
    println("Wrote sum:  $OUT_CSV_SUM")
    return df, df_sum
end

run_genetic_test(INSTANCES_TEST_1000)
