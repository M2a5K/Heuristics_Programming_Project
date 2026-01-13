#!/usr/bin/env julia
using Random

const REPO = normpath(joinpath(@__DIR__, "..", ".."))
include(joinpath(REPO, "MHLibDemos", "src", "SCFPDP.jl"))
include(joinpath(REPO, "MHLibDemos", "test", "run_aco.jl"))  

function (@main)(ARGS)
    instance = ARGS[1]
    seed = parse(Int, ARGS[2])

    num_ants     = 10
    aco_alpha    = 1.0
    aco_beta     = 3.0
    aco_rho      = 0.1
    aco_tau0     = 1.0
    aco_Q        = 1.0
    aco_ls_iters = 0

    i = 3
    while i <= length(ARGS)
        key = ARGS[i]
        val = ARGS[i+1]
        if key == "--num_ants"
            num_ants = parse(Int, val)
        elseif key == "--aco_alpha"
            aco_alpha = parse(Float64, val)
        elseif key == "--aco_beta"
            aco_beta = parse(Float64, val)
        elseif key == "--aco_rho"
            aco_rho = parse(Float64, val)
        elseif key == "--aco_tau0"
            aco_tau0 = parse(Float64, val)
        elseif key == "--aco_Q"
            aco_Q = parse(Float64, val)
        elseif key == "--aco_ls_iters"
            aco_ls_iters = parse(Int, val)
        else
            error("Unknown parameter: $key")
        end
        i += 2
    end

    Random.seed!(seed)

    set_fairness!(:jain)
    sol, iters, runtime = solve_scfpdp(
        "aco", instance;
        seed = seed,
        ttime = 2.0,     
        num_ants = num_ants,
        aco_alpha = aco_alpha,
        aco_beta = aco_beta,
        aco_rho = aco_rho,
        aco_tau0 = aco_tau0,
        aco_Q = aco_Q,
        aco_ls_iters = aco_ls_iters,
        # optionally choose a fixed LS neighborhood 
        # aco_lsparams = LocalSearchParams(:relocate, :first_improvement, false),
    )

    println(sol.obj_val)
end
