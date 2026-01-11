include(joinpath(@__DIR__, "..", "src", "MHLibDemos.jl"))
using .MHLibDemos
using MHLib
using Random

filename = joinpath(@__DIR__, "..", "instances", "50", "test",
                    "instance31_nreq50_nveh2_gamma50.txt")

sol, iters, runtime = MHLibDemos.solve_scfpdp(
    "aco",
    filename;
    seed=1,
    ttime=2.0,
    aco_ttime=2.0,
    num_ants=5,
    aco_alpha=1.0,
    aco_beta=3.0,
    aco_rho=0.1,
    aco_Q=0.2,
    aco_tau0=1.0,

    aco_lsparams=MHLibDemos.LocalSearchParams(:two_opt, :first_improvement, false),
    aco_ls_iters=2,
)

println("iters=$iters runtime=$runtime obj=$(sol.obj_val) feasible=$(MHLibDemos.is_feasible(sol)) served=$(count(sol.served))")
