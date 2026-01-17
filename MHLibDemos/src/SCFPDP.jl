#=
SCF-PDP.jl

Selective Capacitated Fair Pickup and Delivery Problem.

Design fair and feasible routes for a subset of n customer requests. Each customer needs transportation
of a certain amount of goods from a specific pickup location to a corresponding drop-off location.
=#

using Random
using StatsBase
using MHLib

export SCFPDPInstance, SCFPDPSolution, solve_scfpdp

# Fairness measures switch (Task 2 Assignment 2)
const FAIRNESS_MEASURE = Ref{Symbol}(:jain)  # :jain, :maxmin, :gini

"Set fairness measure globally. Use :jain for all later tasks."
set_fairness!(m::Symbol) = (FAIRNESS_MEASURE[] = m)

get_fairness() = FAIRNESS_MEASURE[]



"""
    SCFPDPInstance

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
struct SCFPDPInstance
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


"""
    SCFPDPInstance(file_name)

Read 2D Euclidean SCFPDP instance from file.
"""
function SCFPDPInstance(file_name::AbstractString)
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
        
        return SCFPDPInstance(n, c, gamma, nk, C, rho, depot, pickup, dropoff, coords, d)
    end
end


"""
    SCFPDPSolution

Solution to a SCFPDP instance

Structure:
- `routes`: Vector of routes, each route is a vector of node indices.
- `load`: current load at the end of each insertion (for quick feasibility checks).
- `served`: BitVector of length n (which requests are included).
- `obj_val`: cached objective value.
- `obj_val_valid`: whether the cached objective is valid.
"""
mutable struct SCFPDPSolution <: Solution
    inst::SCFPDPInstance
    routes::Vector{Vector{Int}}     # length nK
    load::Vector{Int}               # length nK
    served::BitVector               # length n
    obj_val::Float64
    obj_val_valid::Bool
end

"""
    SCFPDPSolution(inst::SCFPDPInstance)

Create an empty solution where each vehicle has an empty route starting at the depot.
"""
function SCFPDPSolution(inst::SCFPDPInstance)
    routes = [Int[] for _ in 1:inst.nk]
    load   = fill(0, inst.nk)
    served = falses(inst.n)
    return SCFPDPSolution(inst, routes, load, served, Inf, false)
end

"SCFPDPSolution is minimization."
MHLib.to_maximize(::SCFPDPSolution) = false


"""
    copy!(s1, s2)

Copy solution `s2` into `s1`.
"""
function Base.copy!(s1::SCFPDPSolution, s2::SCFPDPSolution)
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
function Base.copy(s::SCFPDPSolution)
    return SCFPDPSolution(
        s.inst,
        [copy(r) for r in s.routes],
        copy(s.load),
        copy(s.served),
        s.obj_val,
        s.obj_val_valid
    )
end


"""
    Base.show(io, s)

Display the routes and basic info.
"""
function Base.show(io::IO, s::SCFPDPSolution)
    println(io, "SCFPDP Solution:")
    for (k, r) in enumerate(s.routes)
        println(io, " Vehicle $k: ", r)
    end
    println(io, " Served requests: ", findall(s.served))
    println(io, " obj = ", s.obj_val)
end


"""
    calc_objective(::SCFPDPSolution)

Calculates the objective value for the SCF-PDP solution.

The objective consists of:
1. Total travel time across all routes
2. Fairness component based on the Jain fairness measure

Returns the weighted sum: total_time + rho * (1 - fairness)
"""
function MHLib.calc_objective(s::SCFPDPSolution)
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
    # fairness = (total_time ^ 2) / (length(inst.nk) * sum(t^2 for t in route_times))
   
    m = inst.nk  # number of vehicles
    # denom = m * sum(t^2 for t in route_times)
    # fairness = denom == 0.0 ? 0.0 : (total_time ^ 2) / denom

    # Assignment 2 Task 2 
    fairness = fairness_value(route_times)

    # Objective: minimize total time + rho * variance
    return total_time + inst.rho * (1 - fairness)
end

# function MHLib.calc_objective(s::SCFPDPSolution)
#     inst = s.inst

#     # If no requests are served, return a large penalty
#     if !any(s.served)
#         return Inf
#     end

#     # route times
#     route_times = Float64[]
#     total_time = 0.0
#     for route in s.routes
#         t = route_time(inst, route)
#         push!(route_times, t)
#         total_time += t
#     end

#     # No routes → meaningless objective
#     if isempty(route_times)
#         return Inf
#     end

#     # use same definition as in decompose_objective
#     m = inst.nk  # number of vehicles (or use length(route_times))
#     denom = m * sum(t^2 for t in route_times)
#     fairness = denom == 0.0 ? 0.0 : total_time^2 / denom

#     return total_time + inst.rho * (1 - fairness)
# end


"""
    initialize!(s::SCFPDPSolution)

Initialize the solution with empty routes (all vehicles start and end at depot with no requests served).
"""
function MHLib.initialize!(s::SCFPDPSolution)
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

"""
    construct!(s, ::Nothing, result)

`MHMethod` that constructs a new solution by deterministic nearest neighbor construction.
"""
function MHLib.construct!(s::SCFPDPSolution, ::Nothing, result::Result)
    construct_nn_det!(s)
    result.changed = true
end


"""
    LocalSearchParams
Parameters for local search methods in SCF-PDP.
    - `neighborhood`: Symbol indicating the type of neighborhood to use (:two_opt, :inter_route, :relocate)
    - `strategy`: Symbol indicating the improvement strategy (:first_improvement, :best_improvement)
    - `use_delta`: Bool indicating whether to use delta evaluation (true) or full re-evaluation (false)
"""
struct LocalSearchParams
    neighborhood::Symbol  # :two_opt, :inter_route, :relocate
    strategy::Symbol      # :first_improvement, :best_improvement
    use_delta::Bool       # true for delta evaluation, false for full re-evaluation
end

# Keyword constructor with defaults (for easier execution of different parameter configurations in REPL;
# also serves as a default constructor in case no arguments are passed)
LocalSearchParams(; neighborhood=:two_opt, strategy=:first_improvement, use_delta=false) =
    LocalSearchParams(neighborhood, strategy, use_delta)


"""
    local_improve!(s, par, result)

`MHMethod` that performs local search for SCF-PDP.

The `par` parameter specifies the configuration of the local search:
- `par.neighborhood`: Symbol indicating the type of neighborhood to use (:two_opt, :inter_route, :relocate)
- `par.strategy`: Symbol indicating the improvement strategy (:first_improvement, :best_improvement)
- `par.use_delta`: Bool indicating whether to use delta evaluation (true) or full re-evaluation (false)
"""
function MHLib.local_improve!(s::SCFPDPSolution, par::LocalSearchParams, result::Result)
    improved = false

    if par.neighborhood == :two_opt
        improved = local_search_2opt!(s, par)
    elseif par.neighborhood == :inter_route
        improved = local_search_inter_route!(s, par)
    elseif par.neighborhood == :relocate
        improved = local_search_relocate!(s, par)
    else
        error("Unknown neighborhood type: $(par.neighborhood)")
    end

    if improved
        s.obj_val_valid = false  # Force recalculation to be safe
        result.changed = true
    end

    return improved
end

"""
    local_search_2opt!(s, par, result)

Perform 2-opt local search on the SCF-PDP solution `s`.
Returns true if an improving move was applied, false otherwise.
"""
function local_search_2opt!(s::SCFPDPSolution, par::LocalSearchParams)
    improved = false

    # Try 2-opt moves on each vehicle's route
    for k in 1:s.inst.nk
        if par.strategy == :first_improvement
            improved = two_opt_first_improvement!(s, k, par.use_delta)
        elseif par.strategy == :best_improvement
            improved = two_opt_best_improvement!(s, k, par.use_delta)
        end
    end

    return improved
end

"""
    two_opt_first_improvement!(s, k)

Perform first improvement 2-opt on route `k` of solution `s`.
Returns true if an improving move was applied, false otherwise.
"""
function two_opt_first_improvement!(s::SCFPDPSolution, k::Int, use_delta::Bool)
    route = s.routes[k]

    # Need at least 3 nodes for meaningful 2-opt
    if length(route) < 3
        return false
    end

    # Try all pairs of edges to swap
    for i in 1:(length(route)-2)
        for j in (i+2):length(route)
            if use_delta
                # Efficient delta evaluation
                delta = two_opt_delta_eval(s, k, i, j)

                if delta < -1e-6  # Improvement found
                    # Apply the move: reverse segment
                    reverse!(route, i + 1, j)

                    # Check if still feasible (pickup before dropoff)
                    if is_feasible(s)
                        # Accept the move
                        s.obj_val += delta
                        s.obj_val_valid = false
                        return true # Exit after first improvement
                    else
                        # Revert the move
                        reverse!(route, i + 1, j)
                    end
                end
            else
                # Baseline: full objective recalculation
                old_obj = MHLib.obj(s)

                # Apply the move: reverse segment
                reverse!(route, i + 1, j)

                # Check if still feasible (pickup before dropoff)
                if is_feasible(s)
                    # Recalculate full objective
                    s.obj_val_valid = false
                    new_obj = MHLib.obj(s)

                    if new_obj < old_obj - 1e-6  # Improvement found
                        # Accept the move
                        return true # Exit after first improvement
                    else
                        # Revert the move
                        reverse!(route, i + 1, j)
                        s.obj_val = old_obj
                        s.obj_val_valid = true
                    end
                else
                    # Revert the move
                    reverse!(route, i + 1, j)
                    s.obj_val = old_obj
                    s.obj_val_valid = true
                end
            end
        end
    end

    return false
end


"""
    two_opt_best_improvement!(s, k)

Perform best improvement 2-opt on route `k` of solution `s`.
Returns true if an improving move was applied, false otherwise.
"""
function two_opt_best_improvement!(s::SCFPDPSolution, k::Int, use_delta::Bool)
    route = s.routes[k]
    best_move = (Inf, -1, -1) # (obj_val or delta, node1, node2)

    # Need at least 3 nodes for meaningful 2-opt
    if length(route) < 3
        return false
    end

    # Try all pairs of edges to swap
    for i in 1:(length(route)-2)
        for j in (i+2):length(route)
            if use_delta
                # Efficient delta evaluation
                delta = two_opt_delta_eval(s, k, i, j)

                if delta < best_move[1] - 1e-6  # Improvement found
                    # Temporarily apply the move to check feasibility
                    reverse!(route, i + 1, j)

                    # Check if still feasible (pickup before dropoff)
                    if is_feasible(s)
                        best_move = (delta, i, j)
                    end
                    
                    # Revert the move
                    reverse!(route, i + 1, j)
                end
            else
                # Baseline: full objective recalculation
                old_obj = MHLib.obj(s)

                # Apply the move: reverse segment
                reverse!(route, i + 1, j)

                # Check if still feasible (pickup before dropoff)
                if is_feasible(s)
                    # Recalculate full objective
                    s.obj_val_valid = false
                    new_obj = MHLib.obj(s)

                    if new_obj < best_move[1]
                        best_move = (new_obj, i, j)
                    end
                    # Revert the move for now
                    reverse!(route, i + 1, j)
                    s.obj_val = old_obj
                    s.obj_val_valid = true
                else
                    # Revert the move
                    reverse!(route, i + 1, j)
                    s.obj_val = old_obj
                    s.obj_val_valid = true
                end
            end
        end
    end

    # Apply the best move found, if any improvement
    # Check if best move is better than current solution
    current_obj = MHLib.obj(s)
    if best_move[2] != -1 && best_move[1] < current_obj - 1e-6
        _, i_best, j_best = best_move
        reverse!(route, i_best + 1, j_best)
        if use_delta
            s.obj_val += best_move[1]
        else
            s.obj_val = best_move[1]
        end
        s.obj_val_valid = false
        return true
    end

    return false
end


"""
    two_opt_delta_eval(s, k, i, j)

Calculate the change in objective value if we reverse the segment [i+1, j] in route k.

For 2-opt, we replace edges:
- (route[i], route[i+1]) and (route[j], route[j+1])
with:
- (route[i], route[j]) and (route[i+1], route[j+1])

This effectively reverses the segment between i+1 and j.
"""
function two_opt_delta_eval(s::SCFPDPSolution, k::Int, i::Int, j::Int)
    inst = s.inst
    route = s.routes[k]
    
    # Get the four nodes involved
    node_i = (i == 0) ? inst.depot : route[i]
    node_i_next = route[i + 1]
    node_j = route[j]
    node_j_next = (j == length(route)) ? inst.depot : route[j + 1]
    
    # TODO fix calculation of delta: fairness is missing!
    
    # Old edges
    old_dist = inst.d[node_i, node_i_next] + inst.d[node_j, node_j_next]
    
    # New edges (after reversing segment [i+1, j])
    new_dist = inst.d[node_i, node_j] + inst.d[node_i_next, node_j_next]
    
    return new_dist - old_dist
end


"""
    local_search_inter_route!(s, par)

Perform inter-route swapping of two requests on the SCF-PDP solution `s`.
Returns true if an improving move was applied, false otherwise.
"""
function local_search_inter_route!(s::SCFPDPSolution, par::LocalSearchParams)
    improved = false

    # Try inter-route swaps between all pairs of vehicles
    for k1 in 1:(s.inst.nk - 1)
        for k2 in (k1 + 1):s.inst.nk
            if par.strategy == :first_improvement
                improved = inter_route_first_improvement!(s, k1, k2, par.use_delta)
            elseif par.strategy == :best_improvement
                improved = inter_route_best_improvement!(s, k1, k2, par.use_delta)
            end
        end
    end

    return improved
end

"""
    inter_route_first_improvement!(s, k1, k2, use_delta)

Perform first improvement inter-route swapping of two requests between vehicles `k1` and `k2` of solution `s`.
Returns true if an improving move was applied, false otherwise.
"""
function inter_route_first_improvement!(s::SCFPDPSolution, k1::Int, k2::Int, use_delta::Bool)
    route1 = s.routes[k1]
    route2 = s.routes[k2]

    # Try swapping request r1 from k1 with request r2 from k2
    for i1 in 1:(length(route1))
        for i2 in 1:(length(route2))
            r1, is_pickup1 = node_to_request(s.inst, route1[i1])
            r2, is_pickup2 = node_to_request(s.inst, route2[i2])

            # Only consider valid swaps (both pickups; if both are dropoffs, they have already been swapped)
            if is_pickup1 && is_pickup2
                if use_delta
                    # TODO implement inter-route first improvement with delta evaluation
                else
                    # Baseline: full objective recalculation
                    old_obj = MHLib.obj(s)

                    # Get corresponding dropoff nodes for swapping
                    dropoff_node1 = s.inst.dropoff[r1]
                    dropoff_node2 = s.inst.dropoff[r2]
                    idx_dropoff1 = findfirst(==(dropoff_node1), route1)
                    idx_dropoff2 = findfirst(==(dropoff_node2), route2)

                    # Swap the pickup nodes
                    route1[i1], route2[i2] = route2[i2], route1[i1]
                    # Also swap the corresponding dropoff nodes
                    route1[idx_dropoff1], route2[idx_dropoff2] = route2[idx_dropoff2], route1[idx_dropoff1]

                    # Check feasibility
                    if is_feasible(s)
                        # Recalculate full objective
                        s.obj_val_valid = false
                        new_obj = MHLib.obj(s)

                        if new_obj < old_obj - 1e-6  # Improvement found
                            return true # Exit after first improvement
                        else
                            # Revert the swap
                            route1[i1], route2[i2] = route2[i2], route1[i1]
                            route1[idx_dropoff1], route2[idx_dropoff2] = route2[idx_dropoff2], route1[idx_dropoff1]
                            s.obj_val = old_obj
                            s.obj_val_valid = true
                        end
                    else
                        # Revert the swap
                        route1[i1], route2[i2] = route2[i2], route1[i1]
                        route1[idx_dropoff1], route2[idx_dropoff2] = route2[idx_dropoff2], route1[idx_dropoff1]
                        s.obj_val = old_obj
                        s.obj_val_valid = true
                    end
                end
            end
        end
    end
    
    return false
end

"""
    inter_route_best_improvement!(s, k1, k2, use_delta)

Perform best improvement inter-route swapping of two requests between vehicles `k1` and `k2` of solution `s`.
Returns true if an improving move was applied, false otherwise.
"""
function inter_route_best_improvement!(s::SCFPDPSolution, k1::Int, k2::Int, use_delta::Bool)
    route1 = s.routes[k1]
    route2 = s.routes[k2]
    best_move = (Inf, -1, -1, -1, -1) # (obj_val or delta, idx_pickup1, idx_pickup2, idx_dropoff1, idx_dropoff2)

    # Try swapping request r1 from k1 with request r2 from k2
    for i1 in 1:(length(route1))
        for i2 in 1:(length(route2))
            r1, is_pickup1 = node_to_request(s.inst, route1[i1])
            r2, is_pickup2 = node_to_request(s.inst, route2[i2])

            # Only consider valid swaps (both pickups; if both are dropoffs, they have already been swapped)
            if is_pickup1 && is_pickup2
                if use_delta
                    # TODO implement inter-route best improvement with delta evaluation
                else
                    # Baseline: full objective recalculation
                    old_obj = MHLib.obj(s)

                    # Get corresponding dropoff nodes for swapping
                    dropoff_node1 = s.inst.dropoff[r1]
                    dropoff_node2 = s.inst.dropoff[r2]
                    idx_dropoff1 = findfirst(==(dropoff_node1), route1)
                    idx_dropoff2 = findfirst(==(dropoff_node2), route2)

                    # Swap the pickup nodes
                    route1[i1], route2[i2] = route2[i2], route1[i1]
                    # Also swap the corresponding dropoff nodes
                    route1[idx_dropoff1], route2[idx_dropoff2] = route2[idx_dropoff2], route1[idx_dropoff1]

                    # Check feasibility
                    if is_feasible(s)
                        # Recalculate full objective
                        s.obj_val_valid = false
                        new_obj = MHLib.obj(s)

                        if new_obj < best_move[1]
                            best_move = (new_obj, i1, i2, idx_dropoff1, idx_dropoff2)
                        end
                        # Revert the swap for now
                        route1[i1], route2[i2] = route2[i2], route1[i1]
                        route1[idx_dropoff1], route2[idx_dropoff2] = route2[idx_dropoff2], route1[idx_dropoff1]
                        s.obj_val = old_obj
                        s.obj_val_valid = true
                    else
                        # Revert the swap
                        route1[i1], route2[i2] = route2[i2], route1[i1]
                        route1[idx_dropoff1], route2[idx_dropoff2] = route2[idx_dropoff2], route1[idx_dropoff1]
                        s.obj_val = old_obj
                        s.obj_val_valid = true
                    end
                end
            end
        end
    end

    # Apply the best move found, if any improvement
    # Check if best move is better than current solution
    current_obj = MHLib.obj(s)
    if best_move[2] != -1 && best_move[1] < current_obj - 1e-6
        _, i1_best, i2_best, idx_dropoff1_best, idx_dropoff2_best = best_move
        route1[i1_best], route2[i2_best] = route2[i2_best], route1[i1_best]
        route1[idx_dropoff1_best], route2[idx_dropoff2_best] = route2[idx_dropoff2_best], route1[idx_dropoff1_best]
        if use_delta
            s.obj_val += best_move[1]
        else
            s.obj_val = best_move[1]
        end
        s.obj_val_valid = false
        return true
    end

    return false
end


"""
    local_search_relocate!(s, par)

Perform request relocation (moving a request from one vehicle to another) on the SCF-PDP solution `s`.
Returns true if an improving move was applied, false otherwise.
"""
function local_search_relocate!(s::SCFPDPSolution, par::LocalSearchParams)
    improved = false

    # Try relocation from each vehicle to every other vehicle
    for k1 in 1:(s.inst.nk)
        for k2 in 1:s.inst.nk
            if k1 == k2
                continue
            end
            if par.strategy == :first_improvement
                improved = relocate_first_improvement!(s, k1, k2, par.use_delta)
            elseif par.strategy == :best_improvement
                improved = relocate_best_improvement!(s, k1, k2, par.use_delta)
            end
        end
    end

    return improved
end


"""
    relocate_first_improvement!(s, k1, k2, use_delta)

Perform first improvement relocation of a request from vehicle `k1` to vehicle `k2` in solution `s`.
Returns true if an improving move was applied, false otherwise.
"""
function relocate_first_improvement!(s::SCFPDPSolution, k1::Int, k2::Int, use_delta::Bool)
    route1 = s.routes[k1]
    route2 = s.routes[k2]

    for idx_pickup in 1:(length(route1))
        r, is_pickup = node_to_request(s.inst, route1[idx_pickup])
        if !is_pickup
            continue
        else
            dropoff_node = s.inst.dropoff[r]
            idx_dropoff = findfirst(==(dropoff_node), route1)
            # Try all insertion positions in route2
            for pickup_pos in 1:(length(route2) + 1)
                for dropoff_pos in (pickup_pos + 1):(length(route2) + 2)
                    if use_delta
                        # TODO implement relocate first improvement with delta evaluation
                    else
                        # Baseline: full objective recalculation
                        old_obj = MHLib.obj(s)

                        # Remove from route1 (delete higher index first to avoid index shifting issues)
                        pickup_node = route1[idx_pickup]
                        if idx_dropoff > idx_pickup
                            deleteat!(route1, idx_dropoff)
                            deleteat!(route1, idx_pickup)
                        else
                            deleteat!(route1, idx_pickup)
                            deleteat!(route1, idx_dropoff)
                        end

                        # Insert into route2
                        insert!(route2, pickup_pos, pickup_node)
                        insert!(route2, dropoff_pos, dropoff_node)

                        # Check feasibility
                        if is_feasible(s)
                            # Recalculate full objective
                            s.obj_val_valid = false
                            new_obj = MHLib.obj(s)

                            if new_obj < old_obj - 1e-6  # Improvement found
                                return true # Exit after first improvement
                            else
                                # Revert the move (delete from route2 in reverse order)
                                deleteat!(route2, dropoff_pos)
                                deleteat!(route2, pickup_pos)
                                # Insert back into route1 (insert lower index first)
                                if idx_dropoff > idx_pickup
                                    insert!(route1, idx_pickup, pickup_node)
                                    insert!(route1, idx_dropoff, dropoff_node)
                                else
                                    insert!(route1, idx_dropoff, dropoff_node)
                                    insert!(route1, idx_pickup, pickup_node)
                                end
                                s.obj_val = old_obj
                                s.obj_val_valid = true
                            end
                        else
                            # Revert the move (delete from route2 in reverse order)
                            deleteat!(route2, dropoff_pos)
                            deleteat!(route2, pickup_pos)
                            # Insert back into route1 (insert lower index first)
                            if idx_dropoff > idx_pickup
                                insert!(route1, idx_pickup, pickup_node)
                                insert!(route1, idx_dropoff, dropoff_node)
                            else
                                insert!(route1, idx_dropoff, dropoff_node)
                                insert!(route1, idx_pickup, pickup_node)
                            end
                            s.obj_val = old_obj
                            s.obj_val_valid = true
                        end
                    end
                end
            end
        end
    end
    
    return false
end


"""
    relocate_best_improvement!(s, k1, k2, use_delta)

Perform best improvement relocation of a request from vehicle `k1` to vehicle `k2` in solution `s`.
Returns true if an improving move was applied, false otherwise.
"""
function relocate_best_improvement!(s::SCFPDPSolution, k1::Int, k2::Int, use_delta::Bool)
    route1 = s.routes[k1]
    route2 = s.routes[k2]
    best_move = (Inf, -1, -1, -1, -1)
    # idx_pickup and idx_dropoff refer to positions in route1
    # pickup_pos and dropoff_pos refer to positions in route2

    for idx_pickup in 1:(length(route1))
        r, is_pickup = node_to_request(s.inst, route1[idx_pickup])
        if !is_pickup
            continue
        else
            pickup_node = route1[idx_pickup]
            dropoff_node = s.inst.dropoff[r]
            idx_dropoff = findfirst(==(dropoff_node), route1)
            # Try all insertion positions in route2
            for pickup_pos in 1:(length(route2) + 1)
                for dropoff_pos in (pickup_pos + 1):(length(route2) + 2)
                    if use_delta
                        # TODO implement relocate first improvement with delta evaluation
                    else
                        # Baseline: full objective recalculation
                        old_obj = MHLib.obj(s)

                        # Remove from route1 (delete higher index first to avoid index shifting issues)
                        if idx_dropoff > idx_pickup
                            deleteat!(route1, idx_dropoff)
                            deleteat!(route1, idx_pickup)
                        else
                            deleteat!(route1, idx_pickup)
                            deleteat!(route1, idx_dropoff)
                        end

                        # Insert into route2
                        insert!(route2, pickup_pos, pickup_node)
                        insert!(route2, dropoff_pos, dropoff_node)

                        # Check feasibility
                        if is_feasible(s)
                            # Recalculate full objective
                            s.obj_val_valid = false
                            new_obj = MHLib.obj(s)

                            if new_obj < best_move[1]  # Improvement found
                                best_move = (new_obj, idx_pickup, idx_dropoff, pickup_pos, dropoff_pos)
                            end
                            # Revert the move for now
                            deleteat!(route2, dropoff_pos)
                            deleteat!(route2, pickup_pos)
                            # Insert back into route1 (insert lower index first)
                            if idx_dropoff > idx_pickup
                                insert!(route1, idx_pickup, pickup_node)
                                insert!(route1, idx_dropoff, dropoff_node)
                            else
                                insert!(route1, idx_dropoff, dropoff_node)
                                insert!(route1, idx_pickup, pickup_node)
                            end
                            s.obj_val = old_obj
                            s.obj_val_valid = true
                        else
                            # Revert the move (delete from route2 in reverse order)
                            deleteat!(route2, dropoff_pos)
                            deleteat!(route2, pickup_pos)
                            # Insert back into route1 (insert lower index first)
                            if idx_dropoff > idx_pickup
                                insert!(route1, idx_pickup, pickup_node)
                                insert!(route1, idx_dropoff, dropoff_node)
                            else
                                insert!(route1, idx_dropoff, dropoff_node)
                                insert!(route1, idx_pickup, pickup_node)
                            end
                            s.obj_val = old_obj
                            s.obj_val_valid = true
                        end
                    end
                end
            end
        end
    end

    # Apply the best move found, if any improvement
    # Check if best move is better than current solution
    current_obj = MHLib.obj(s)
    if best_move[2] != -1 && best_move[1] < current_obj - 1e-6
        _, idx_pickup_best, idx_dropoff_best, pickup_pos_best, dropoff_pos_best = best_move

        # Remove from route1 (delete higher index first to avoid index shifting issues)
        pickup_node = route1[idx_pickup_best]
        dropoff_node = route1[idx_dropoff_best]
        if idx_dropoff_best > idx_pickup_best
            deleteat!(route1, idx_dropoff_best)
            deleteat!(route1, idx_pickup_best)
        else
            deleteat!(route1, idx_pickup_best)
            deleteat!(route1, idx_dropoff_best)
        end

        # Insert into route2
        insert!(route2, pickup_pos_best, pickup_node)
        insert!(route2, dropoff_pos_best, dropoff_node)

        if use_delta
            s.obj_val += best_move[1]
        else
            s.obj_val = best_move[1]
        end
        s.obj_val_valid = false
        return true
    end

    return false
end


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




# HELPER FUNCTIONS 
"""
node_to_request(inst, node) -> (r, is_pickup)

Map a node index to the corresponding request index and a flag that
tells you whether it is a pickup node.

Returns:
- `(r, true)`  if `node` is the pickup of request `r`
- `(r, false)` if `node` is the dropoff of request `r`
- `(0, true)`  if `node` is the depot (no request)
"""
function node_to_request(inst::SCFPDPInstance, node::Int)
    if node == inst.depot
        return 0, true
    elseif 2 <= node <= inst.n + 1
        # pickups are 2..(n+1)
        return node - 1, true
    elseif (inst.n + 2) <= node <= (2 * inst.n + 1)
        # dropoffs are (n+2)..(2n+1)
        return node - (inst.n + 1), false
    else
        error("node_to_request: node $node out of range")
    end
end


"""
route_time(inst, route) -> Int

Compute the total travel time of a single route, including:
- from depot to the first node (if any),
- between all consecutive nodes,
- from the last node back to the depot.

If the route is empty, the time is 0.
"""
function route_time(inst::SCFPDPInstance, route::Vector{Int})
    if isempty(route)
        return 0
    end

    t = 0
    # depot to first
    t += inst.d[inst.depot, route[1]]

    # between customers
    for i in 1:(length(route) - 1)
        t += inst.d[route[i], route[i+1]]
    end

    # last to depot
    t += inst.d[route[end], inst.depot]

    return t
end


"""
extra_cost_if_append(inst, route, r) -> Int

Compute how much the travel time of `route` increases if we append
request `r` at the end as [pickup(r), dropoff(r)].

This simple version just recomputes the route time before and after.
"""
function extra_cost_if_append(inst::SCFPDPInstance,
                              route::Vector{Int},
                              r::Int)
    p = inst.pickup[r]
    q = inst.dropoff[r]

    old_time = route_time(inst, route)

    # temporarily extended route
    tmp = copy(route)
    push!(tmp, p)
    push!(tmp, q)

    new_time = route_time(inst, tmp)
    return new_time - old_time
end


"""
append_request!(s, k, r)

Append request `r` at the end of vehicle `k`'s route as
[pickup(r), dropoff(r)] and update `served`.

We *do not* touch `s.obj_val` here – that is done by `calc_objective`.
"""
function append_request!(s::SCFPDPSolution, k::Int, r::Int)
    inst = s.inst
    push!(s.routes[k], inst.pickup[r])
    push!(s.routes[k], inst.dropoff[r])
    s.served[r] = true
    return s
end


"""
is_feasible(s) -> Bool

Check basic feasibility of a solution:

1. Capacity constraints:
   - the load never exceeds vehicle capacity
   - the load never becomes negative
2. Precedence:
   - dropoff of request r cannot appear before its pickup

We recompute the load along each route from scratch.
"""
function is_feasible(s::SCFPDPSolution)
    inst = s.inst

    for route in s.routes
        load = 0
        seen_pickup = falses(inst.n)  

        for node in route
            r, is_pickup = node_to_request(inst, node)

            if is_pickup
                load += inst.c[r]
                seen_pickup[r] = true
            else
                # dropoff must come after pickup
                if !seen_pickup[r]
                    return false
                end
                load -= inst.c[r]
            end

            if load < 0 || load > inst.C
                return false
            end
        end
    end

    return true
end

# Extension of the above feasibility check that also checks
# whether at least gamma requests are served.
function is_feasible_with_gamma_check(s::SCFPDPSolution)
    inst = s.inst

    for route in s.routes
        load = 0
        seen_pickup = falses(inst.n)  

        for node in route
            r, is_pickup = node_to_request(inst, node)

            if is_pickup
                load += inst.c[r]
                seen_pickup[r] = true
            else
                # dropoff must come after pickup
                if !seen_pickup[r]
                    return false
                end
                load -= inst.c[r]
            end

            if load < 0 || load > inst.C
                return false
            end
        end
    end

    # check gamma, which is the number of requests that must be served
    if count(s.served) < inst.gamma
        return false
    end

    return true
end


"""
Assignment 2 Task 2 
Compute fairness F in [0,1] from per-vehicle route times d(R_k).
We interpret d(R_k) as the travel time of route k incl. depot legs.
"""
function fairness_value(route_times::Vector{Float64}; measure::Symbol = FAIRNESS_MEASURE[])
    m = length(route_times)
    m == 0 && return 0.0

    if measure == :jain
        s = sum(route_times)
        denom = m * sum(t^2 for t in route_times)
        return denom == 0.0 ? 0.0 : (s^2) / denom
    elseif measure == :maxmin
        mx = maximum(route_times)
        mn = minimum(route_times)
        return mx <= 0.0 ? 0.0 : mn / mx
    # elseif measure == :maxmin
    #     eps = 1e-9
    #     mx = maximum(route_times)
    #     mn = minimum(route_times)
    #     return mx <= 0.0 ? 0.0 : (mn + eps) / (mx + eps)

    elseif measure == :gini
        s = sum(route_times)
        s <= 0.0 && return 0.0
        num = 0.0
        @inbounds for i in 1:m
            xi = route_times[i]
            for j in 1:m
                num += abs(xi - route_times[j])
            end
        end
        denom = 2.0 * m * s
        g = 1.0 - (num / denom)
        return clamp(g, 0.0, 1.0)
    else
        error("Unknown fairness measure: $measure (use :jain, :maxmin, :gini)")
    end
end

"Compute per-vehicle route times vector [d(R_1), ..., d(R_|K|)]."
function route_times_vec(s::SCFPDPSolution)
    inst = s.inst
    rt = Float64[]
    for route in s.routes
        push!(rt, float(route_time(inst, route)))
    end
    return rt
end

"Stops per route (counts nodes; pickup and dropoff are each a stop)."
stops_per_route(s::SCFPDPSolution) = [length(r) for r in s.routes]




"""
    construct_nn_det!(s)

Deterministic greedy construction heuristic for SCF-PDP.

Starting from an empty solution, it repeatedly chooses the request/vehicle
pair (r, k) that yields the smallest increase in total travel time when
request r is appended at the end of vehicle k's route as
[pickup(r), dropoff(r)], subject to feasibility (capacity + precedence).

The process stops when either:
- `gamma` requests have been served, or
- no further feasible insertion can be found.
"""
function construct_nn_det!(s::SCFPDPSolution)
    inst = s.inst
    # empty solution
    initialize!(s)

    # initial objective value to compare against
    best_obj_val = s.obj_val

    # all requests initially available
    remaining = collect(1:inst.n)

    while count(s.served) < inst.gamma && !isempty(remaining)
        best_r   = 0
        best_k   = 0
        best_Δ   = Inf

        # Try to insert each remaining request into each vehicle
        for r in remaining
            for k in 1:inst.nk
                # apply the insertion on a temporary copy for now
                tmp = copy(s)
                append_request!(tmp, k, r)
                new_obj_val = MHLib.calc_objective(tmp)
                Δ = new_obj_val - best_obj_val

                # test feasibility and check cost increase if we appended r to route k
                if is_feasible(tmp) && Δ < best_Δ
                    best_Δ = Δ
                    best_r = r
                    best_k = k
                    best_obj_val = new_obj_val
                end
            end
        end

        if best_r == 0
            break
        end

        # Commit the best insertion to the real solution
        append_request!(s, best_k, best_r)
        deleteat!(remaining, findfirst(==(best_r), remaining))
    end

    # comp objective
    s.obj_val = MHLib.calc_objective(s)
    s.obj_val_valid = true
    return s
end


"""
    construct_nn_rand!(s; alpha=0.3)

Randomized greedy construction heuristic for SCF-PDP.

This is a GRASP-style variant of the deterministic nearest-neighbor construction.
At each step, we build a candidate list (CL) of all feasible request–vehicle pairs,
evaluate their marginal cost Δ(k,r), and derive a Restricted Candidate List (RCL)
containing only sufficiently good candidates. One element of the RCL is then chosen
uniformly at random and applied.

The parameter `alpha ∈ [0,1]` controls the level of randomization:
- `alpha = 0` reproduces the purely greedy construction.
- `alpha = 1` allows any feasible insertion to be chosen.
"""
function construct_nn_rand!(s::SCFPDPSolution; alpha::Float64 = 0.3)
    inst = s.inst
    initialize!(s)

    # initial objective value to compare against
    best_obj_val = s.obj_val

    remaining = collect(1:inst.n)

    while count(s.served) < inst.gamma && !isempty(remaining)
        # candidate list 
        # each element Δ, r, k
        candidates = Tuple{Float64,Int,Int}[]

        for r in remaining
            for k in 1:inst.nk
                # apply the insertion on a temporary copy for now
                tmp = copy(s)
                append_request!(tmp, k, r)
                new_obj_val = MHLib.calc_objective(tmp)
                Δ = new_obj_val - best_obj_val

                # insert to check feasibility
                tmp = copy(s)
                append_request!(tmp, k, r)

                if is_feasible(tmp)
                    push!(candidates, (Δ, r, k))
                end
            end
        end

        isempty(candidates) && break

        # parameters for our formula 
        Δ_min = minimum(c[1] for c in candidates)
        Δ_max = maximum(c[1] for c in candidates)

        # lecture notes formula , alpha can be determined by user
        # RCL = { (Δ,r,k) in CL | Δ <= Δ_min + alpha*(Δ_max - Δ_min) }
        thresh = Δ_min + alpha * (Δ_max - Δ_min)
        rcl = [(Δ, r, k) for (Δ, r, k) in candidates if Δ <= thresh]

        # safety: if alpha is tiny and RCL gets empty, fall back to all candidates
        isempty(rcl) && (rcl = candidates)

        # random selection from RCL
        chosen = rand(rcl)
        _, r_star, k_star = chosen

        # apply the chosen insertion to the real solution
        append_request!(s, k_star, r_star)
        deleteat!(remaining, findfirst(==(r_star), remaining))
    end

    # finalize objective
    s.obj_val = MHLib.calc_objective(s)
    s.obj_val_valid = true
    return s
end

# do the same but over multipl iters and return best 
function multistart_randomized_construction(inst, titer, scheduler_kwargs; alpha=0.3, iters=50)
    best = nothing
    best_val = Inf

    # collect multiple scheduling runs; their metrics will be used in solve_scfpdp
    heuristic = Vector{GVNS}(undef, iters)

    for i in 1:iters
        s = SCFPDPSolution(inst)
        heuristic[i] = GVNS(s, [MHMethod("con", (s, _, r) -> (construct_nn_rand!(s; alpha=alpha); r.changed = true))],
            MHMethod[],
            MHMethod[];
            consider_initial_sol = false,
            titer = titer,
            scheduler_kwargs...,
        )

        if s.obj_val < best_val
            best = copy(s)
            best_val = s.obj_val
        end

        run!(heuristic[i])
    end

    return heuristic
end



"""
    greedy_complete!(s)

Greedy completion of a partial SCF-PDP solution.

Starting from the current `s` (with some requests already served), this function
repeatedly inserts the feasible request–vehicle pair with the smallest marginal
increase in route time, until `γ` requests are served or no feasible insertion
remains. This is essentially the deterministic NN construction, but starting
from a non-empty solution.
"""
function greedy_complete!(s::SCFPDPSolution)
    inst = s.inst
    remaining = findall(!, s.served)  # requests not yet served

    # initial objective value to compare against
    best_obj_val = s.obj_val

    while count(s.served) < inst.gamma && !isempty(remaining)
        best_Δ = Inf
        best_r = 0
        best_k = 0

        for r in remaining
            for k in 1:inst.nk
                # apply the insertion on a temporary copy for now
                tmp = copy(s)
                append_request!(tmp, k, r)
                new_obj_val = MHLib.calc_objective(tmp)
                Δ = new_obj_val - best_obj_val

                if is_feasible(tmp) && Δ < best_Δ
                    best_Δ = Δ
                    best_r = r
                    best_k = k
                    best_obj_val = new_obj_val
                end
            end
        end

        best_r == 0 && break

        append_request!(s, best_k, best_r)
        deleteat!(remaining, findfirst(==(best_r), remaining))
    end

    s.obj_val = MHLib.calc_objective(s)
    s.obj_val_valid = true
    return s
end


"""
    construct_pilot!(s; iters_per_step=1)

Pilot construction heuristic for SCF-PDP.

At each step, for every feasible request–vehicle insertion (k,r), we simulate the
completion of the partial solution with a greedy heuristic (`greedy_complete!`)
and select the insertion whose completed solution has the best objective value.
"""
function construct_pilot!(s::SCFPDPSolution)
    inst = s.inst
    initialize!(s)

    remaining = collect(1:inst.n)

    while count(s.served) < inst.gamma && !isempty(remaining)
        best_val = Inf
        best_r   = 0
        best_k   = 0

        for r in remaining
            for k in 1:inst.nk
                # insert
                tmp = copy(s)
                append_request!(tmp, k, r)

                # must be feasible before completion
                is_feasible(tmp) || continue

                # complete greedily from here
                greedy_complete!(tmp)

                if tmp.obj_val < best_val
                    best_val = tmp.obj_val
                    best_r   = r
                    best_k   = k
                end
            end
        end

        best_r == 0 && break

        # commit the best pilot move
        append_request!(s, best_k, best_r)
        deleteat!(remaining, findfirst(==(best_r), remaining))
    end

    s.obj_val = MHLib.calc_objective(s)
    s.obj_val_valid = true
    return s
end

# following TSP random_two_exchange_moves!.
"""
    random_shake!(s, par)

Simple shaking for GVNS:
try up to `par` random relocate moves between vehicles
to perturb the current solution.
"""
function random_shake!(s::SCFPDPSolution, par::Int)
    inst = s.inst

    for _ in 1:par
        # Work on a temporary copy; only commit if feasible
        tmp = copy(s)

        # 1) Choose a source route with at least one pickup
        k1_candidates = Int[]
        for k in 1:inst.nk
            route = tmp.routes[k]
            # route must contain at least one pickup node (non-depot)
            has_pickup = any(begin
                r, is_pickup = node_to_request(inst, node)
                is_pickup && r != 0
            end for node in route)
            has_pickup && push!(k1_candidates, k)
        end
        isempty(k1_candidates) && continue

        k1 = rand(k1_candidates)
        route1 = tmp.routes[k1]

        # 2) Pick a random pickup in route1
        pickup_indices = Int[]
        for (idx, node) in enumerate(route1)
            r, is_pickup = node_to_request(inst, node)
            is_pickup && r != 0 && push!(pickup_indices, idx)
        end
        isempty(pickup_indices) && continue

        idx_pickup = rand(pickup_indices)
        r, _ = node_to_request(inst, route1[idx_pickup])
        pickup_node  = inst.pickup[r]
        dropoff_node = inst.dropoff[r]

        idx_dropoff = findfirst(==(dropoff_node), route1)
        idx_dropoff === nothing && continue

        # 3) Remove pickup+dropoff from route1 (on tmp)
        # Always delete the higher index first to avoid shifting issues
        if idx_dropoff > idx_pickup
            deleteat!(route1, idx_dropoff)
            deleteat!(route1, idx_pickup)
        else
            deleteat!(route1, idx_pickup)
            deleteat!(route1, idx_dropoff)
        end

        # 4) Choose target route (can be same or different)
        k2 = rand(1:inst.nk)
        route2 = tmp.routes[k2]

        # Sample insertion positions based on CURRENT length of route2
        len2 = length(route2)
        # pickup can go anywhere 1..len2+1
        pickup_pos  = rand(1:len2 + 1)
        # dropoff must come after pickup, so in [pickup_pos+1 .. len2+2]
        dropoff_pos = rand(pickup_pos + 1:len2 + 2)

        # 5) Insert into target route
        insert!(route2, pickup_pos,  pickup_node)
        insert!(route2, dropoff_pos, dropoff_node)

        # 6) If shaken solution is feasible → keep it, else discard
        if is_feasible(tmp)
            copy!(s, tmp)
            s.obj_val_valid = false
        end
        # if infeasible, do nothing (we just ignore this shake)
    end

    return s
end


"""
    shaking!(s, par, result)

GVNS shaking wrapper for SCFPDP.
"""
function MHLib.shaking!(s::SCFPDPSolution, par::Int, result::Result)
    random_shake!(s, par)
    result.changed = true
end



# Include ACO and Genetic algorithms (requires the following to be defined at this point:
# SCFPDPInstance, SCFPDPSolution, LocalSearchParams, and the associated MHLib methods)
include("ACO.jl")
include("Genetic.jl")


"""
    solve_scfpdp(alg::AbstractString, filename::AbstractString; seed=nothing, titer=1000, 
        kwargs...)

Solve a given SCFPDP instance with the algorithm `alg`.

# Parameters
- `filename`: File name of the SCFPDP instance
- `alg`: Algorithm to apply ("nn_det", "nn_rand", "pilot", "ls", "vnd", "grasp")
- `seed`: Possible random seed for reproducibility; if `nothing`, a random seed is chosen
- `titer`: Number of iterations for the solving algorithm, gets a new default value
- `kwargs`: Additional configuration parameters passed to the algorithm, e.g., `ttime`
"""
# function solve_scfpdp(alg::AbstractString = "nn_det",
#         filename::AbstractString = joinpath(dirname(dirname(pathof(MHLibDemos))),
#                                             "instances", "50", "train",
#                                             "instance1_nreq50_nveh2_gamma50.txt");
#         seed = nothing, titer = 1000, kwargs...)
function solve_scfpdp(alg::AbstractString = "nn_det",
                      filename::AbstractString = "";
                      seed = nothing,
                      titer::Int = 1000,
                      lsparams::LocalSearchParams = LocalSearchParams(),
                      initsol = nothing,
                      kwargs...)

    # require filename when using this helper
    if filename == ""
        error("solve_scfpdp: keyword `filename` must be provided")
    end

    isnothing(seed) && (seed = rand(0:typemax(Int32)))
    Random.seed!(seed)

    # override fields in lsparams from kwargs if provided
    if :neighborhood in keys(kwargs)
        lsparams = LocalSearchParams(
            neighborhood=kwargs[:neighborhood],
            strategy=lsparams.strategy,
            use_delta=lsparams.use_delta
        )
    end
    if :strategy in keys(kwargs)
        lsparams = LocalSearchParams(
            neighborhood=lsparams.neighborhood,
            strategy=kwargs[:strategy],
            use_delta=lsparams.use_delta
        )
    end
    if :use_delta in keys(kwargs)
        lsparams = LocalSearchParams(
            neighborhood=lsparams.neighborhood,
            strategy=lsparams.strategy,
            use_delta=kwargs[:use_delta]
        )
    end

    # Filter kwargs to only include valid SchedulerConfig parameters
    # to avoid "ERROR: MethodError: no method matching MHLib.SchedulerConfig" when passing lsparams as kwargs to solve_scfpdp
    valid_scheduler_keys = (:checkit, :log, :lnewinc, :lfreq, :ttime, :tciter, :tctime, :tobj)
    scheduler_kwargs = filter(p -> p.first in valid_scheduler_keys, kwargs)

    println("SCFPDP solver called with parameters:")
    println("alg=$alg, filename=$filename, seed=$seed, ", (; kwargs...))

    inst = SCFPDPInstance(filename)
    sol  = SCFPDPSolution(inst)

    if alg == "nn_det"
        heuristic = GVNS(
            sol,
            [MHMethod("con", (s, _, r) -> (construct_nn_det!(s); r.changed = true))],
            MHMethod[],
            MHMethod[];
            consider_initial_sol = false,
            titer = titer,
            scheduler_kwargs...,
        )

    elseif alg == "nn_rand"
        alpha = get(kwargs, :alpha, 0.3)
        heuristic = GVNS(sol, [MHMethod("con", (s, _, r) -> (construct_nn_rand!(s; alpha=alpha); r.changed = true))],
            MHMethod[],
            MHMethod[];
            consider_initial_sol = false,
            titer = titer,
            scheduler_kwargs...,
        )

    elseif alg == "nn_rand_multi"
        iters = get(kwargs, :iters, 50)
        alpha = get(kwargs, :alpha, 0.3)
        heuristic = multistart_randomized_construction(inst, titer, scheduler_kwargs; alpha = alpha, iters = iters)

    elseif alg == "pilot"
        heuristic = GVNS(
            sol,
            [MHMethod("con", (s, _, r) -> (construct_pilot!(s); r.changed = true))],
            MHMethod[],
            MHMethod[];
            consider_initial_sol = false,
            titer = titer,
            scheduler_kwargs...,
        )

    elseif alg == "ls"
        if initsol !== nothing
            copy!(sol, initsol)
            heuristic = GVNS(
                sol,
                MHMethod[],
                [MHMethod("li1", local_improve!,
                    LocalSearchParams(lsparams.neighborhood, lsparams.strategy, lsparams.use_delta))],
                MHMethod[];
                consider_initial_sol = true,
                titer = titer,
                scheduler_kwargs...,
            )
        else
            heuristic = GVNS(
                sol,
                [MHMethod("con", construct!)],
                [MHMethod("li1", local_improve!,
                    LocalSearchParams(lsparams.neighborhood, lsparams.strategy, lsparams.use_delta))],
                MHMethod[];
                consider_initial_sol = true,
                titer = titer,
                scheduler_kwargs...,
            )
        end

    elseif alg == "vnd"
        if initsol !== nothing
            copy!(sol, initsol)
            heuristic = GVNS(
                sol,
                MHMethod[],
                [
                    MHMethod("li1", local_improve!,
                        LocalSearchParams(:relocate, lsparams.strategy, lsparams.use_delta)),
                    MHMethod("li2", local_improve!,
                        LocalSearchParams(:inter_route, lsparams.strategy, lsparams.use_delta)),
                    MHMethod("li3", local_improve!,
                        LocalSearchParams(:two_opt, lsparams.strategy, lsparams.use_delta)),
                ],
                MHMethod[];
                consider_initial_sol = true,
                titer = titer,
                scheduler_kwargs...,
            )
        else
            heuristic = GVNS(
                sol,
                [MHMethod("con", construct!)],
                [
                    MHMethod("li1", local_improve!,
                        LocalSearchParams(:relocate, lsparams.strategy, lsparams.use_delta)),
                    MHMethod("li2", local_improve!,
                        LocalSearchParams(:inter_route, lsparams.strategy, lsparams.use_delta)),
                    MHMethod("li3", local_improve!,
                        LocalSearchParams(:two_opt, lsparams.strategy, lsparams.use_delta)),
                ],
                MHMethod[];
                consider_initial_sol = true,
                titer = titer,
                scheduler_kwargs...,
            )
        end

    elseif alg == "gen_vns"
        if initsol !== nothing
            copy!(sol, initsol)
            heuristic = GVNS(
                sol,
                MHMethod[],
                [
                    MHMethod("li1", local_improve!,
                        LocalSearchParams(:relocate, lsparams.strategy, lsparams.use_delta)),
                    MHMethod("li2", local_improve!,
                        LocalSearchParams(:inter_route, lsparams.strategy, lsparams.use_delta)),
                    MHMethod("li3", local_improve!,
                        LocalSearchParams(:two_opt, lsparams.strategy, lsparams.use_delta)),
                ],
                [MHMethod("sh1", shaking!, 1)];
    
                consider_initial_sol = true,
                titer = titer,
                scheduler_kwargs...,
            )
        else
            heuristic = GVNS(
                sol,
                [MHMethod("con", construct!)],
                [
                    MHMethod("li1", local_improve!,
                        LocalSearchParams(:relocate, lsparams.strategy, lsparams.use_delta)),
                    MHMethod("li2", local_improve!,
                        LocalSearchParams(:inter_route, lsparams.strategy, lsparams.use_delta)),
                    MHMethod("li3", local_improve!,
                        LocalSearchParams(:two_opt, lsparams.strategy, lsparams.use_delta)),
                ],
                # MHMethod[];
                [MHMethod("sh1", shaking!, 1)];

                consider_initial_sol = true,
                titer = titer,
                scheduler_kwargs...,
            )
        end


    # ASSIGNMENT PART2: solving with ACO 
    elseif alg == "aco"
    num_ants = get(kwargs, :num_ants, 5)
    aco_alpha = get(kwargs, :aco_alpha, 1.0)
    aco_beta  = get(kwargs, :aco_beta, 3.0)
    aco_rho   = get(kwargs, :aco_rho, 0.1)
    aco_tau0  = get(kwargs, :aco_tau0, 1.0)
    aco_Q     = get(kwargs, :aco_Q, 1.0)
    aco_seed  = get(kwargs, :aco_seed, seed)  

    aco_ttime = get(kwargs, :aco_ttime, get(kwargs, :ttime, 2.0))

    # optional: local search inside ACO
    aco_ls_iters = get(kwargs, :aco_ls_iters, 0)
    aco_lsparams = get(kwargs, :aco_lsparams, nothing)  # pass a LocalSearchParams or nothing

    heuristic = GVNS(
        sol,
        [MHMethod("aco_con",
            (s, _, r) -> begin
            MHLib.initialize!(s) 
            s.obj_val_valid = false

                run_aco!(s;
                    num_ants=num_ants,
                    alpha=aco_alpha,
                    beta=aco_beta,
                    rho=aco_rho,
                    ttime=aco_ttime,
                    seed=aco_seed,
                    tau0=aco_tau0,
                    Q=aco_Q,
                    lsparams=aco_lsparams,
                    ls_iters=aco_ls_iters
                )
                s.obj_val = MHLib.calc_objective(s)
                s.obj_val_valid = true
                r.changed = true
            end
        )],
        MHMethod[],   # no extra local search here (can be added)
        MHMethod[];
        consider_initial_sol = false,
        titer = 1,     # important: ACO already loops internally; 1 call is enough and it always print 1 iter 
        scheduler_kwargs...,
    )


    # ASSIGNMENT PART2: solving with Genetic Algorithm 
    elseif alg == "genetic"
    pop_size = get(kwargs, :pop_size, 30)
    crossover_rate = get(kwargs, :crossover_rate, 0.9)
    mutation_rate = get(kwargs, :mutation_rate, 0.2)
    tournament_size = get(kwargs, :tournament_size, 3)
    elite_size = get(kwargs, :elite_size, 2)
    genetic_ttime = get(kwargs, :ttime, get(kwargs, :ttime, 2.0))
    max_generations = get(kwargs, :max_generations, 1000)
    constr_mthd = get(kwargs, :construction_method, Symbol("nn_rand"))  # or "pilot" or "nn_det"
    genetic_seed = get(kwargs, :ga_seed, seed)
    verbose = get(kwargs, :verbose, false)

    heuristic = GVNS(
        sol,
        [MHMethod("genetic_con",
            (s, _, r) -> begin
            MHLib.initialize!(s) 
            s.obj_val_valid = false

                run_genetic!(s;
                    pop_size=pop_size,
                    crossover_rate=crossover_rate,
                    mutation_rate=mutation_rate,
                    tournament_size=tournament_size,
                    elite_size=elite_size,
                    ttime=genetic_ttime,
                    max_generations=max_generations,
                    construction_method=constr_mthd,
                    seed=genetic_seed,
                    verbose=verbose
                )
                s.obj_val = MHLib.calc_objective(s)
                s.obj_val_valid = true
                r.changed = true
            end
        )],
        MHMethod[],
        MHMethod[];
        consider_initial_sol = false,
        titer = 1,     # Genetic Algorithm already loops internally, therefore 1 call is enough here
        scheduler_kwargs...,
    ) 


    elseif alg == "grasp"
        niters = get(kwargs, :niters, 20)   # how many GRASP iterations
        alpha  = get(kwargs, :alpha, 0.3)   # same alpha as in construct_nn_rand!
        heuristic = Vector{GVNS}(undef, niters)

        # Local search parameters
        ls_par = LocalSearchParams(lsparams.neighborhood, lsparams.strategy, lsparams.use_delta)

        best_sol = nothing
        best_val = Inf

        for it in 1:niters
            s_it = SCFPDPSolution(inst)
            
            # Create GVNS with construction and local improvement
            heuristic[it] = GVNS(
                s_it,
                [MHMethod("con", (s, _, r) -> (construct_nn_rand!(s; alpha=alpha); r.changed = true))],
                [MHMethod("li_grasp", local_improve!, ls_par)],
                MHMethod[];
                consider_initial_sol = false,
                titer = titer,
                scheduler_kwargs...,
            )

            # Run construction and local improvement through the framework
            run!(heuristic[it])

            val = MHLib.obj(s_it)
            if val < best_val
                best_val = val
                best_sol = copy(s_it)
            end
        end

        if best_sol === nothing
            error("GRASP failed to generate any solution")
        end

        copy!(sol, best_sol)

    else
        error("Algorithm '$alg' not yet implemented.")
    end


    if alg != "nn_rand_multi" && alg != "grasp"
        run!(heuristic)
        copy!(sol, heuristic.scheduler.incumbent)
    else
        # nn_rand_multi already ran inside its function
        for heur in heuristic
            if heur.scheduler.incumbent.obj_val < sol.obj_val
                copy!(sol, heur.scheduler.incumbent)
            end
        end
    end

    total_iterations = 0
    total_runtime = 0.0
    # it is possible that 'heuristic' is either a vector (nn_rand_multi) or a single GVNS
    heuristic = heuristic isa AbstractVector ? heuristic : [heuristic]
    for heur in heuristic
        method_statistics(heur.scheduler)
        main_results(heur.scheduler)
        println("Feasible? ", is_feasible(heur.scheduler.incumbent))
        total_iterations += heur.scheduler.iteration
        total_runtime += heur.scheduler.run_time
    end

    # TODO no sure whether to average (in case of nn_rand_multi)
    # if we do not average, runtimes are too high (program finishes in 0.5s, but total_time says 25s bc of 50 runs)
    total_iterations /= length(heuristic)
    total_runtime /= length(heuristic)

    save_solution(sol, alg, filename, seed, titer, lsparams)

    return sol, total_iterations, total_runtime
end


# To run from REPL, activate `MHLibDemos` environment, use `MHLibDemos`,
# and call e.g. `solve_scfpdp("ls", titer=200, seed=1)`.

# Run with profiler:
# @profview solve_scfpdp(args)



"""
    decompose_objective(s) into (total_time, fairness, obj)

Recompute the SCF-PDP objective and return its components:
- total_time = sum of route times
- fairness   = Jain-type fairness term
- obj        = total_time + ρ * fairness
"""
# function decompose_objective(s::SCFPDPSolution)
#     inst = s.inst

#     # route times
#     route_times = Float64[]
#     total_time = 0.0
#     for route in s.routes
#         t = route_time(inst, route)
#         push!(route_times, t)
#         total_time += t
#     end

#     if isempty(route_times)
#         return (Inf, 0.0, Inf)
#     end

#     fairness = (total_time^2) / (length(inst.nk) * sum(t^2 for t in route_times))
#     obj = total_time + inst.rho * (1 - fairness)

#     return (total_time, fairness, obj)
# end

function decompose_objective(s::SCFPDPSolution)
    inst = s.inst

    # route times per vehicle
    route_times = Float64[]
    total_time = 0.0
    for route in s.routes
        t = route_time(inst, route)
        push!(route_times, t)
        total_time += t
    end

    # No routes → meaningless objective
    if isempty(route_times)
        return (Inf, 0.0, Inf)
    end

    # Number of vehicles (or use length(route_times) if you prefer only active ones)
    m = inst.nk

    # denom = m * sum(t^2 for t in route_times)
    # fairness = denom == 0.0 ? 0.0 : total_time^2 / denom

    # Asignment 2 Task 2
    fairness = fairness_value(route_times)



    obj = total_time + inst.rho * (1 - fairness)

    return (total_time, fairness, obj)
end




"""
    save_solution(s::SCFPDPSolution, alg::String, filename::AbstractString,
                       seed::Int, titer::Int, lsparams::LocalSearchParams)
Save the SCFPDP solution `s` to a file with metadata in the filename.
"""
function save_solution(s::SCFPDPSolution, alg::String, filename::AbstractString,
                       seed::Int, titer::Int, lsparams::LocalSearchParams)
    outdir = joinpath(@__DIR__, "..", "test", "results", "solutions")
    mkpath(outdir)

    instancename = splitext(basename(filename))[1]
    outfile = joinpath(outdir,
        "sol_$(instancename)_$(alg)_s$(seed)_t$(titer)" *
        "_$(lsparams.neighborhood)_$(lsparams.strategy)_$(lsparams.use_delta).txt")

    open(outfile, "w") do io
        println(io, instancename)
        for (k, r) in enumerate(s.routes)
            println(io, join(r, " "))
        end
    end

    println("Solution saved to: $outfile")
end
