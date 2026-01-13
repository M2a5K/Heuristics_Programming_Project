# # julia --project=MHLibDemos MHLibDemos/test/run_task2_fairness.jl

# include(joinpath(@__DIR__, "..", "src", "MHLibDemos.jl"))
# using .MHLibDemos

# filename = joinpath(@__DIR__, "..", "instances", "50", "test",
#                     "instance31_nreq50_nveh2_gamma50.txt")

# function run_one(measure::Symbol)
#     MHLibDemos.set_fairness!(measure)

#     sol, iters, runtime = MHLibDemos.solve_scfpdp(
#         "aco",
#         filename;
#         seed=1,
#         ttime=2.0,
#         aco_ttime=2.0,
#         num_ants=5,
#         aco_alpha=1.0,
#         aco_beta=3.0,
#         aco_rho=0.25,
#         aco_Q=0.2,
#         aco_tau0=1.0,
#         aco_lsparams=MHLibDemos.LocalSearchParams(:two_opt, :first_improvement, false),
#         aco_ls_iters=2,
#     )

#     rt = MHLibDemos.route_times_vec(sol)
#     stops = MHLibDemos.stops_per_route(sol)
#     fair = MHLibDemos.fairness_value(rt; measure=measure)

#     println("\n=== fairness=$measure ===")
#     println("obj=$(sol.obj_val) runtime=$(runtime) iters=$(iters)")
#     println("feasible=$(MHLibDemos.is_feasible(sol)) served=$(count(sol.served))")
#     println("route_times=$(rt) fairness=$(fair)")
#     println("stops_per_route=$(stops) total_stops=$(sum(stops))")

#     return sol
# end

# run_one(:jain)
# run_one(:maxmin)
# run_one(:gini)

# MHLibDemos.set_fairness!(:jain)
# println("\nReverted fairness to $(MHLibDemos.get_fairness()).")



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

base = joinpath(@__DIR__, "..", "instances", "50", "test")
instances = [
    "instance31_nreq50_nveh2_gamma50.txt",
    "instance32_nreq50_nveh2_gamma45.txt",
    "instance33_nreq50_nveh2_gamma45.txt",
    "instance34_nreq50_nveh2_gamma47.txt",
    "instance45_nreq50_nveh2_gamma45.txt",
]
seeds = 1:5
fairnesses = [:jain, :maxmin, :gini]

outdir = joinpath(@__DIR__, "results")
mkpath(outdir)
outfile = joinpath(outdir, "fairness_experiment_$(Dates.format(now(), "yyyymmdd_HHMM")).csv")

open(outfile, "w") do io
    println(io, "instance,seed,fairness,obj,feasible,served,iters,runtime,route_times,stops_per_route,total_stops,fairness_value")
    for instfile in instances
        filename = joinpath(base, instfile)
        for seed in seeds, F in fairnesses
            @printf("Running %s seed=%d fairness=%s...\n", instfile, seed, String(F))
            res = run_one(filename; seed=seed, fairness=F)
            println(io,
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

println("Saved: ", outfile)
