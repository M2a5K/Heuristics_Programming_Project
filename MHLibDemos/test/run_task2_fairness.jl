using Random
using Printf
using Dates

include(joinpath(@__DIR__, "..", "src", "MHLibDemos.jl"))
using .MHLibDemos
using MHLib

function route_times_and_stops(sol::MHLibDemos.SCFPDPSolution)
    inst = sol.inst
    rt = Float64[]
    st = Int[]
    for k in 1:inst.nk
        r = sol.routes[k]
        push!(st, length(r))
        push!(rt, MHLibDemos.route_time(inst, r))
    end
    return rt, st
end

# run one configuration 
function run_one(filename::String; seed::Int, fairness::Symbol)
    MHLibDemos.set_fairness!(fairness)

    sol, iters, runtime = MHLibDemos.solve_scfpdp(
        "aco", filename;
        seed=seed,
        ttime=2.0,
        aco_ttime=2.0,
        num_ants=5,
        aco_alpha=1.0,
        aco_beta=3.0,
        aco_rho=0.25,
        aco_Q=0.2,
        aco_tau0=1.0,
        aco_lsparams=MHLibDemos.LocalSearchParams(:two_opt, :first_improvement, false),
        aco_ls_iters=2,
    )

    rt, st = route_times_and_stops(sol)
    # fair = MHLibDemos.fairness_value(rt, fairness) 
    fair = MHLibDemos.fairness_value(rt; measure=fairness)
 
    feasible = MHLibDemos.is_feasible(sol)
    served = count(sol.served)

    MHLibDemos.set_fairness!(:jain)

    return (obj=sol.obj_val, iters=iters, runtime=runtime,
            feasible=feasible, served=served,
            rt=rt, st=st, fairness=fair)
end


const INSTANCES_TEST_50 = [
    "instance31_nreq50_nveh2_gamma50.txt",
    "instance32_nreq50_nveh2_gamma45.txt",
    "instance33_nreq50_nveh2_gamma45.txt",
    "instance34_nreq50_nveh2_gamma47.txt",
    "instance45_nreq50_nveh2_gamma45.txt",
]

const INSTANCES_TEST_100 = [
    "instance31_nreq100_nveh2_gamma86.txt",
    "instance36_nreq100_nveh2_gamma90.txt",
    "instance41_nreq100_nveh2_gamma86.txt",
    "instance46_nreq100_nveh2_gamma92.txt",
    "instance51_nreq100_nveh2_gamma93.txt",
    "instance56_nreq100_nveh2_gamma89.txt",
]

const INSTANCES_TEST_200 = [
    "instance31_nreq200_nveh4_gamma192.txt",
    "instance36_nreq200_nveh4_gamma181.txt",
    "instance41_nreq200_nveh4_gamma196.txt",
    "instance46_nreq200_nveh4_gamma178.txt",
    "instance51_nreq200_nveh4_gamma172.txt",
    "instance56_nreq200_nveh4_gamma171.txt",
]

instance_sets = [
    (50,  joinpath(@__DIR__, "..", "instances", "50",  "test"), INSTANCES_TEST_50),
    (100, joinpath(@__DIR__, "..", "instances", "100", "test"), INSTANCES_TEST_100),
    (200, joinpath(@__DIR__, "..", "instances", "200", "test"), INSTANCES_TEST_200),
]

seeds = 1:5
fairnesses = [:jain, :maxmin, :gini]

outdir = joinpath(@__DIR__, "results")
mkpath(outdir)
outfile = joinpath(outdir, "fairness_experiment_$(Dates.format(now(), "yyyymmdd_HHMM")).csv")

open(outfile, "w") do io
    println(io, "n,instance,seed,fairness,obj,feasible,served,iters,runtime,route_times,stops_per_route,total_stops,fairness_value")

    for (n, base, instfiles) in instance_sets
        for instfile in instfiles
            filename = joinpath(base, instfile)
            if !isfile(filename)
                @warn "Missing instance file, skipping" filename
                continue
            end

            for seed in seeds, F in fairnesses
                @printf("Running n=%d %s seed=%d fairness=%s...\n", n, instfile, seed, String(F))
                res = run_one(filename; seed=seed, fairness=F)

                println(io,
                    n, ",",
                    string(instfile), ",", seed, ",", F, ",",
                    res.obj, ",", res.feasible, ",", res.served, ",",
                    res.iters, ",", res.runtime, ",",
                    "\"", res.rt, "\",",
                    "\"", res.st, "\",",
                    sum(res.st), ",",
                    res.fairness
                )
            end
        end
    end
end

println("Saved: ", outfile)
