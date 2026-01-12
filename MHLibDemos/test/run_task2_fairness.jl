# julia --project=MHLibDemos MHLibDemos/test/run_task2_fairness.jl

include(joinpath(@__DIR__, "..", "src", "MHLibDemos.jl"))
using .MHLibDemos

filename = joinpath(@__DIR__, "..", "instances", "50", "test",
                    "instance31_nreq50_nveh2_gamma50.txt")

function run_one(measure::Symbol)
    MHLibDemos.set_fairness!(measure)

    sol, iters, runtime = MHLibDemos.solve_scfpdp(
        "aco",
        filename;
        seed=1,
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

    rt = MHLibDemos.route_times_vec(sol)
    stops = MHLibDemos.stops_per_route(sol)
    fair = MHLibDemos.fairness_value(rt; measure=measure)

    println("\n=== fairness=$measure ===")
    println("obj=$(sol.obj_val) runtime=$(runtime) iters=$(iters)")
    println("feasible=$(MHLibDemos.is_feasible(sol)) served=$(count(sol.served))")
    println("route_times=$(rt) fairness=$(fair)")
    println("stops_per_route=$(stops) total_stops=$(sum(stops))")

    return sol
end

run_one(:jain)
run_one(:maxmin)
run_one(:gini)

MHLibDemos.set_fairness!(:jain)
println("\nReverted fairness to $(MHLibDemos.get_fairness()).")
