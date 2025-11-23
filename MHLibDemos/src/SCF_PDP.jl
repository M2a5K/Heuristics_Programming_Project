#=
SCF-PDP.jl

Selective Capacitated Fair Pickup and Delivery Problem.

Design fair and feasible routes for a subset of n customer requests. Each customer needs transportation
of a certain amount of goods from a specific pickup location to a corresponding drop-off location.
=#

using Random
using StatsBase
using MHLib

export SCF_PDP_Instance, SCF_PDP_Solution, solve_scf_pdp


"""
    SCF_PDP_Instance

Selective Capacitated Fair Pickup and Delivery Problem (SCF-PDP) instance.

Design fair and feasible routes for a subset of n customer requests. Each customer needs transportation
of a certain amount of goods from a specific pickup location to a corresponding drop-off location.

# Attributes
- `n`: number of customer requests
- `c`: amount of goods to be transported for each customer request
- `gamma`: number of requests that must be fulfilled
- `nk`: number of vehicles
- `C`: capacity of each vehicle
- `rho`: weighting parameter that controls the trade-off between total travel time and fairness
- `depot`: index of vehicle depot
- `pickup`: indices of pickup locations
- `dropoff`: indices of drop-off locations
- `coords`: Euclidean coordinates of locations (including vehicle depot)
- `d`: distance matrix
"""
struct SCF_PDP_Instance
    n::Int
    c::Vector{Int}
    gamma::Int
    nk::Int
    C::Int
    rho::Float64
    
    depot::Int
    pickup::Vector{Int}
    dropoff::Vector{Int}

    coords::Vector{Tuple{Int,Int}}

    d::Matrix{Int}
end


# TODO remove this function
# """
#     SCF_PDP_Instance(coords, c, gamma, nk, C, rho)

# Create a SCF_PDP instance from given Euclidean coordinates and other parameters specific to the problem.
# """
# function SCF_PDP_Instance(coords::Vector{Tuple{Int,Int}},
#     c::Vector{Int}, gamma::Int, nk::Int, C::Int, rho::Float64)

#     n = length(c)
#     @assert length(coords) == length(c) * 2 + 1
#     d = Matrix{Int}(undef, n+1, n+1)
#     for i in 1:(n+1) # also count the depot
#         for j in i:n
#             p = coords[i]; q = coords[j]
#              if i == j
#                  d[i,j] = 0
#              else
#                  d[i,j] = d[j,i] = round(Int, sqrt((p[1] - q[1])^2 + (p[2] - q[2])^2))
#              end
#         end
#     end
#     return SCF_PDP_Instance(n, c, gamma, nk, C, rho, d, coords_depot,
#         coords_pickup, coords_dropoff)
# end

"""
    SCF_PDP_Instance(file_name)

Read 2D Euclidean SCF_PDP instance from file.
"""
function SCF_PDP_Instance(file_name::AbstractString)
    open(file_name) do f
        n, nk, C, gamma, rho = readline(f) |> strip |> split |> x -> 
            (parse(Int, x[1]), parse(Int, x[2]), parse(Int, x[3]),
            parse(Int, x[4]), parse(Float64, x[5]))

        # demands
        line = readline(f)
        @assert startswith(line, "# demands")
        c = parse.(Int, split(readline(f)))

        # coordinates
        line = readline(f)
        @assert startswith(lowercase(line), "# request")

        lines = readlines(f)
        coords = Tuple{Int,Int}[]
        for line in lines
            stripped = strip(line)
            isempty(stripped) && continue
            xy = parse.(Int, split(stripped))
            push!(coords, (xy[1], xy[2]))
        end

        # indices
        depot = 1
        pickup = 2:(n + 1)
        dropoff = (n + 2):(2*n + 1)

        # distance matrix
        m = length(coords)
        d = Matrix{Int}(undef, m, m)
        for i in 1:m
            xi, yi = coords[i]
            for j in i:m
                xj, yj = coords[j]
                d[i, j] = d[j, i] = (i == j ? 0 :
                                     ceil(Int, sqrt((xi - xj)^2 + (yi - yj)^2)))
            end
        end

        length(coords) == 0 && error("No coordinates found in file $file_name")
        (length(coords) - 1)/2 != length(c) && 
            error("Number of request locations and demand values do not match in file $file_name")
        
        return SCF_PDP_Instance(n, c, gamma, nk, C, rho, depot, pickup, dropoff, coords, d)
    end
end

# """
#     TSPInstance(n, dims::Vector=[100, 100])

# Create a random Euclidean TSP instance with `n` nodes.

# The nodes lie in the integer grid `[0, xdim-1] x [0, ydim-1]`.
# """
# function TSPInstance(n::Int=50, dims::Vector=[100, 100])
#     @assert length(dims) == 2
#     coords = [trunc.(Int, rand(2) .* dims) for _ in 1:n]
#     TSPInstance(coords)
# end

# function Base.show(io::IO, inst::TSPInstance)
#     println(io, "n=$(inst.n), d=$(inst.d)")
# end


"""
    SCF_PDP_Solution

Solution to a SCF_PDP instance

Structure:
- `routes`: Vector of routes, each route is a vector of node indices.
- `load`: current load at the end of each insertion (for quick feasibility checks).
- `served`: BitVector of length n (which requests are included).
- `obj_val`: cached objective value.
- `obj_val_valid`: whether the cached objective is valid.
"""
mutable struct SCF_PDP_Solution <: Solution
    inst::SCF_PDP_Instance
    routes::Vector{Vector{Int}}     # length nK
    load::Vector{Int}               # length nK
    served::BitVector               # length n
    obj_val::Float64
    obj_val_valid::Bool
end

"""
    SCF_PDP_Solution(inst::SCF_PDP_Instance)

Create an empty solution where each vehicle has an empty route starting at the depot.
"""
function SCF_PDP_Solution(inst::SCF_PDP_Instance)
    routes = [Int[] for _ in 1:inst.nk]
    load   = fill(0, inst.nk)
    served = falses(inst.n)
    return SCF_PDP_Solution(inst, routes, load, served, Inf, false)
end

"SCF_PDP_Solution is minimization."
MHLib.to_maximize(::SCF_PDP_Solution) = false

# SCF_PDP_Solution(inst::SCF_PDP_Instance) =
#     SCF_PDP_Solution(inst, Vector{Int}[collect(1:inst.nk)], collect(1:inst.nk), Vector{Bool}(undef, inst.n), Inf, false)

"""
    copy!(s1, s2)

Copy solution `s2` into `s1`.
"""
function Base.copy!(s1::SCF_PDP_Solution, s2::SCF_PDP_Solution)
    s1.inst = s2.inst
    s1.routes = [copy(r) for r in s2.routes]
    s1.load   = copy(s2.load)
    s1.served = copy(s2.served)
    s1.obj_val = s2.obj_val
    s1.obj_val_valid = s2.obj_val_valid

    return s1
end

"""
    copy(s::SCFPDPSolution)

Deep copy constructor.
"""
function Base.copy(s::SCF_PDP_Solution)
    return SCF_PDP_Solution(
        s.inst,
        [copy(r) for r in s.routes],
        copy(s.load),
        copy(s.served),
        s.obj_val,
        s.obj_val_valid
    )
end

# Base.copy(s::TSPSolution) =
#     TSPSolution(s.inst, s.obj_val, s.obj_val_valid, copy(s.x), 
#         (isnothing(s.destroyed) ? nothing : copy(s.destroyed)))

"""
    Base.show(io, s)

Display the routes and basic info.
"""
function Base.show(io::IO, s::SCF_PDP_Solution)
    println(io, "SCF_PDP Solution:")
    for (k, r) in enumerate(s.routes)
        println(io, " Vehicle $k: ", r)
    end
    println(io, " Served requests: ", findall(s.served))
    println(io, " obj = ", s.obj_val)
end


"""
    calc_objective(::SCF_PDP_Solution)

Calculates the objective value for the SCF-PDP solution.

The objective consists of:
1. Total travel time across all routes
2. Fairness component based on the Jain fairness measure

Returns the weighted sum: total_time + rho * variance
"""
function MHLib.calc_objective(s::SCF_PDP_Solution)
    inst = s.inst
    
    # If no requests are served, return a large penalty
    if !any(s.served)
        return Inf
    end
    
    # Calculate travel time for each route
    route_times = Float64[]
    total_time = 0.0
    
    for (k, route) in enumerate(s.routes)
        if isempty(route)
            push!(route_times, 0.0)
            continue
        end
        
        # Calculate time for this route: depot -> route[1] -> ... -> route[end] -> depot
        time = inst.d[inst.depot, route[1]]  # depot to first node
        
        for i in 1:(length(route)-1)
            time += inst.d[route[i], route[i+1]]  # between consecutive nodes
        end
        
        time += inst.d[route[end], inst.depot]  # last node back to depot
        
        push!(route_times, time)
        total_time += time
    end
    
    # Calculate fairness component using the Jain fairness measure
    # TODO Do we have to exclude unused routes? According to the handout, we sum over all K, where K is the
    # "fleet of nK identical vehicles". Is this interpreted as all vehicles, or only the used ones?
    # used_routes = filter(t -> t > 0, route_times)

    fairness = (total_time ^ 2) / (length(s.routes) * sum(t^2 for t in route_times))
    
    # Objective: minimize total time + rho * variance
    return total_time + inst.rho * fairness
end

"""
    initialize!(s::SCF_PDP_Solution)

Initialize the solution with empty routes (all vehicles start and end at depot with no requests served).
"""
function MHLib.initialize!(s::SCF_PDP_Solution)
    # Reset all routes to empty
    for k in 1:s.inst.nk
        empty!(s.routes[k])
        s.load[k] = 0
    end
    
    # Mark all requests as not served
    fill!(s.served, false)
    
    # Set objective value to Inf (no requests served means infeasible/very poor solution)
    s.obj_val = Inf
    s.obj_val_valid = false
    
    return s
end

# """
#     construct!(tsp_solution, ::Nothing, result)

# `MHMethod` that constructs a new solution by random initialization.
# """
# MHLib.construct!(s::TSPSolution, ::Nothing, result::Result) = initialize!(s)

# """
#     local_improve!(tsp_solution, ::Any, result)

# `MHMethod` that performs two-opt local search.
# """
# function MHLib.local_improve!(s::TSPSolution, ::Any, result::Result)
#     if !two_opt_neighborhood_search!(s, false)
#         result.changed = false
#     end
# end

# """
#     shaking!(tsp_solution, par, result)

# `MHMethod` that performs shaking by making `par` random 2-exchange move.
# """
# function MHLib.shaking!(s::TSPSolution, par::Int, result::Result)
#     random_two_exchange_moves!(s, par)
# end

# """
#     destroy!(tsp_solution, par, result)

# `MHMethod` that Performs a destroy operation by removing nodes from the solution.

# The number of removed nodes is `3 * par`.
# """
# function MHLib.destroy!(s::TSPSolution, par::Int, ::Result)
#     random_remove_elements!(s, get_number_to_destroy(s, length(s.x); 
#         min_abs=3par, max_abs=3par))
# end

# """
#     repair!(tsp_solution, ::Nothing, result)
    
# `MHMethod` that performs a repair by reinserting removed nodes randomly.
# """
# MHLib.repair!(s::TSPSolution, ::Nothing, ::Result) = greedy_reinsert_removed!(s)

# """
#     insert_val_at_best_pos!(tsp_solution, val)

# Inserts `val` greedily at the best position.
# The solution's objective value is assumed to be valid and is incrementally updated.
# """
# function MHLib.insert_val_at_best_pos!(s::TSPSolution, val::Int)
#     x = s.x
#     d = s.inst.d
#     best_pos = length(x) + 1
#     δ_best = δ = d[val, x[end]] + d[val, x[1]] - d[x[1], x[end]]
#     for i in 2:length(s.x)
#         δ = d[val, x[i-1]] + d[val, x[i]] - d[x[i-1], x[i]]       
#         if δ < δ_best 
#             δ_best = δ
#             best_pos = i
#         end
#     end
#     insert!(s.x, best_pos, val)
#     s.obj_val = s.obj_val + δ_best
# end

# """
#     two_opt_move_delta_eval(permutation_solution, p1, p2)

# Return efficiently the delta in the objective value when 2-opt move would be applied.
# """
# function MHLib.two_opt_move_delta_eval(s::TSPSolution, p1::Integer, 
#         p2::Integer)
#     @assert 1 <= p1 < p2 <= length(s)
#     if p1 == 1 && p2 == length(s)
#         # reversing the whole solution has no effect
#         return 0
#     end
#     prev = mod1(p1 - 1, length(s))
#     nxt = mod1(p2 + 1, length(s))

#     x_p1 = s.x[p1]
#     x_p2 = s.x[p2]
#     x_prev = s.x[prev]
#     x_next = s.x[nxt]
#     delta = s.inst.d[x_prev,x_p2] + s.inst.d[x_p1,x_next] - s.inst.d[x_prev,x_p1] - 
#         s.inst.d[x_p2,x_next]
# end


# # -------------------------------------------------------------------------------

"""
    solve_scf_pdp(alg::AbstractString, filename::AbstractString; seed=nothing, titer=1000, 
        kwargs...)

Solve a given SCF_PDP instance with the algorithm `alg`.

# Parameters
- `filename`: File name of the SCF_PDP instance
- `alg`: Algorithm to apply ("nn_det", "nn_rand", "pilot", "beam", "vnd", "grasp")
- `seed`: Possible random seed for reproducibility; if `nothing`, a random seed is chosen
- `titer`: Number of iterations for the solving algorithm, gets a new default value
- `kwargs`: Additional configuration parameters passed to the algorithm, e.g., `ttime`
"""
function solve_scf_pdp(alg::AbstractString="nn_det",
        filename::AbstractString=joinpath(@__DIR__, "..", "instances", "50", "train", "instance1_nreq50_nveh2_gamma50.txt");
        seed=nothing, titer=1000, kwargs...)
    # Make results reproducibly by either setting a given seed or picking one randomly
    isnothing(seed) && (seed = rand(0:typemax(Int32)))
    Random.seed!(seed)

    println("SCF_PDP solver called with parameters:")
    println("alg=$alg, filename=$filename, seed=$seed, ", (; kwargs...))

    inst = SCF_PDP_Instance(filename)
    sol = SCF_PDP_Solution(inst)
    initialize!(sol)
    println(sol)

    # TODO this has not been adapted to SCF_PDP yet (still TSP baseline)
    # if alg === "lns"
    #     heuristic = LNS(sol, MHMethod[MHMethod("con", construct!)],
    #         [MHMethod("de$i", destroy!, i) for i in 1:3],
    #         [MHMethod("re", repair!)]; 
    #         consider_initial_sol=true, titer, kwargs...)
    # elseif alg === "gvns"
    #     heuristic = GVNS(sol, [MHMethod("con", construct!)],
    #         [MHMethod("li1", local_improve!, 1)], [MHMethod("sh1", shaking!, 1)];
    #         consider_initial_sol=true, titer, kwargs...)
    # else
    #     error("Invalid parameter alg: $alg")
    # end
    # run!(heuristic)
    # method_statistics(heuristic.scheduler)
    # main_results(heuristic.scheduler)
    # check(sol)
    return sol
end

# To run from REPL, activate `MHLibDemos` environment, use `MHLibDemos`,
# and call e.g. `solve_tsp("lns", titer=200, seed=1)`.

# Run with profiler:
# @profview solve_tsp(args)
