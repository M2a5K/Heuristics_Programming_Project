# Run search-heuristic experiments for several instances and α-values.
#
# IMPORTANT: start Julia with one thread, e.g.
#   JULIA_NUM_THREADS=1 julia

using MHLibDemos
using MHLibDemos: decompose_objective
using Random
using DataFrames
using CSV
using Glob

function run_search_experiments(; 
        rootdir  = joinpath(@__DIR__, "..", "instances", "50", "train"),
        inst_files = [
            "instance1_nreq50_nveh2_gamma50.txt",
            "instance3_nreq50_nveh2_gamma49.txt",
            "instance7_nreq50_nveh2_gamma41.txt",
            "instance14_nreq50_nveh2_gamma45.txt",
            "instance27_nreq50_nveh2_gamma41.txt",
        ],
        seeds   = 1:10)

    searchparams = [ [:two_opt, :inter_route, :relocate],
                     [:first_improvement, :best_improvement],
                     [false] ]
    # exclude delta evaluations for now, as it is not implemented for all neighborhoods

    rows = NamedTuple[]

    for inst_file in inst_files
        filename_inst = joinpath(rootdir, inst_file)
        inst_name_clean = replace(inst_file, ".txt" => "")
        # filename_sol = joinpath(@__DIR__, "results", "solutions", "sol_$(inst_name_clean)_nn_det*.txt")

        # Check if solution file exists and load it
        sol_files = glob("sol_$(inst_name_clean)_nn_det*.txt", joinpath(@__DIR__, "results", "solutions"))
        if !isempty(sol_files)
            filename_sol = first(sol_files)
            inst = SCFPDPInstance(filename_inst)
            init_sol = read_solution(filename_sol, inst, "train")
        else
            inst = SCFPDPInstance(filename_inst)
            init_sol = SCFPDPSolution(inst)
            construct_nn_det!(init_sol)
        end

        for seed in seeds
            # --- local search ---
            for neighborhood in searchparams[1]
                for strategy in searchparams[2]
                    for use_delta in searchparams[3]
                        ls_params = MHLibDemos.LocalSearchParams(neighborhood, strategy, use_delta)
                        sol, iters, time = solve_scfpdp("ls", filename_inst; seed, lsparams=ls_params, initsol=init_sol)
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
            for strategy in searchparams[2]
                for use_delta in searchparams[3]
                    # the neighborhood does not matter here, because VND uses all of them;
                    # still, we need to provide something to make the LocalSearchParams struct happy
                    ls_params = MHLibDemos.LocalSearchParams(:two_opt, strategy, use_delta)
                    sol, iters, time = solve_scfpdp("vnd", filename_inst; seed, lsparams=ls_params)
                    t, fair, obj = decompose_objective(sol)
                    inst = sol.inst  # same instance, but nice and explicit

                    push!(rows, (; 
                        instance        = inst_file,
                        n               = inst.n,
                        nk              = inst.nk,
                        gamma           = inst.gamma,
                        C               = inst.C,
                        rho             = inst.rho,
                        alg             = "vnd",
                        seed            = seed,
                        neighborhood    = :all,
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
    end

    df = DataFrame(rows)

    outdir = joinpath(@__DIR__, "results")
    mkpath(outdir)

    outfile = joinpath(outdir, "search_results_experiments.csv")
    CSV.write(outfile, df)

    println("Results saved to: $outfile")
    println(first(df, 10))  # show a small preview
end


function read_solution(filename::String, inst::SCFPDPInstance, type::String="train")
    sol = SCFPDPSolution(inst)
    open(filename, "r") do io
        # first line points to instance file; parse the values found there
        header = readline(io)
        instance_name = strip(header)
        nreq_match = match(r"nreq(\d+)", instance_name)
        nreq = nreq_match.captures[1]
        file_inst = joinpath(@__DIR__, "..", "instances", nreq, type, instance_name * ".txt")
        inst = SCFPDPInstance(file_inst)
        sol = SCFPDPSolution(inst)
        for (i, line) in enumerate(eachline(io))
            if i == 1
                continue
            end
            sol.routes[i-1] = parse.(Int, split(line))
        end
    end
    return sol
end


# run_search_experiments()
