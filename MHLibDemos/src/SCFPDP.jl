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

    # Only consider used routes (non-zero times) for fairness calculation to be correct.
    # According to the handout, we sum over all K, where K is the "fleet of nK identical vehicles".
    # We interpret this as all the vehicles that are actually being used.
    used_routes = filter(t -> t > 0, route_times)
    
    if isempty(used_routes)
        fairness = 0.0
    else
        fairness = (total_time ^ 2) / (length(used_routes) * sum(t^2 for t in used_routes))
    end
    
    # Objective: minimize total time + rho * variance
    return total_time + inst.rho * (1 - fairness)
end

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
    construct_nn_rand!(s)
    result.changed = true
end


"""
    LocalSearchParams
Parameters for local search methods in SCF-PDP.
    - `neighborhood`: Symbol indicating the type of neighborhood to use (:two_opt, :three_opt, :inter_route)
    - `strategy`: Symbol indicating the improvement strategy (:first_improvement, :best_improvement)
    - `use_delta`: Bool indicating whether to use delta evaluation (true) or full re-evaluation (false)
"""
struct LocalSearchParams
    neighborhood::Symbol  # :two_opt, :three_opt, :inter_route
    strategy::Symbol      # :first_improvement, :best_improvement
    use_delta::Bool       # true for delta evaluation, false for full re-evaluation
end

# Default constructor for backward compatibility
LocalSearchParams() = LocalSearchParams(:two_opt, :first_improvement, false)


"""
    local_improve!(s, par, result)

`MHMethod` that performs local search for SCF-PDP.

The `par` parameter specifies the configuration of the local search:
- `par.neighborhood`: Symbol indicating the type of neighborhood to use (:two_opt, :three_opt)
- `par.strategy`: Symbol indicating the improvement strategy (:first_improvement, :best_improvement)
- `par.use_delta`: Bool indicating whether to use delta evaluation (true) or full re-evaluation (false)
"""
function MHLib.local_improve!(s::SCFPDPSolution, par::LocalSearchParams, result::Result)
    improved = false

    if par.neighborhood == :two_opt
        improved = local_search_2opt!(s, par)
    elseif par.neighborhood == :three_opt
        improved = local_search_3opt!(s, par)
    elseif par.neighborhood == :inter_route
        improved = local_search_inter_route!(s, par)
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

    # Apply the best move found, if any
    if best_move[2] != -1
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

    # Try inter-route moves between all pairs of vehicles
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
    inter_route_first_improvement!(s, k1, k2)

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
                            s.obj_val_valid = true
                        end
                    else
                        # Revert the swap
                        route1[i1], route2[i2] = route2[i2], route1[i1]
                        route1[idx_dropoff1], route2[idx_dropoff2] = route2[idx_dropoff2], route1[idx_dropoff1]
                        s.obj_val_valid = true
                    end
                end
            end
        end
    end
    
    return false
end

"""
    inter_route_best_improvement!(s, k1, k2)

Perform best improvement inter-route swapping of two requests between vehicles `k1` and `k2` of solution `s`.
Returns true if an improving move was applied, false otherwise.
"""
function inter_route_best_improvement!(s::SCFPDPSolution, k1::Int, k2::Int, use_delta::Bool)
    # TODO implement inter-route best improvement
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
            # TODO this check might be unnecessary, because we don't store the depot in a route
            if r == 0
                continue  # depot should not appear inside routes
            end

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

    # all requests initially available
    remaining = collect(1:inst.n)

    while count(s.served) < inst.gamma && !isempty(remaining)
        best_r   = 0
        best_k   = 0
        best_Δ   = Inf

        # Try to insert each remaining request into each vehicle
        for r in remaining
            for k in 1:inst.nk
                # check cost increase if we appended r to route k
                Δ = extra_cost_if_append(inst, s.routes[k], r)

                # apply the insertion and test feasibility
                tmp = copy(s)
                append_request!(tmp, k, r)

                if is_feasible(tmp) && Δ < best_Δ
                    best_Δ = Δ
                    best_r = r
                    best_k = k
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

    remaining = collect(1:inst.n)

    while count(s.served) < inst.gamma && !isempty(remaining)
        # andidate list 
        # each element Δ, r, k
        candidates = Tuple{Float64,Int,Int}[]

        for r in remaining
            for k in 1:inst.nk
                # extra cost if we append r to route k
                Δ = extra_cost_if_append(inst, s.routes[k], r)

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

        # lectre notes formula , alpha can be dtermined by user
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
function multistart_randomized_construction(inst; alpha=0.3, iters=50)
    best = nothing
    best_val = Inf

    for _ in 1:iters
        s = SCFPDPSolution(inst)
        construct_nn_rand!(s; alpha)

        if s.obj_val < best_val
            best = copy(s)
            best_val = s.obj_val
        end
    end

    return best
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

    while count(s.served) < inst.gamma && !isempty(remaining)
        best_Δ = Inf
        best_r = 0
        best_k = 0

        for r in remaining
            for k in 1:inst.nk
                Δ = extra_cost_if_append(inst, s.routes[k], r)

                tmp = copy(s)
                append_request!(tmp, k, r)

                if is_feasible(tmp) && Δ < best_Δ
                    best_Δ = Δ
                    best_r = r
                    best_k = k
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




"""
    solve_scfpdp(alg::AbstractString, filename::AbstractString; seed=nothing, titer=1000, 
        kwargs...)

Solve a given SCFPDP instance with the algorithm `alg`.

# Parameters
- `filename`: File name of the SCFPDP instance
- `alg`: Algorithm to apply ("nn_det", "nn_rand", "pilot", "beam", "ls", "vnd", "grasp")
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
                      kwargs...)

    # require filename when using this helper
    if filename == ""
        error("solve_scfpdp: keyword `filename` must be provided")
    end

    isnothing(seed) && (seed = rand(0:typemax(Int32)))
    Random.seed!(seed)

    println("SCFPDP solver called with parameters:")
    println("alg=$alg, filename=$filename, seed=$seed, ", (; kwargs...))

    inst = SCFPDPInstance(filename)
    sol  = SCFPDPSolution(inst)

    if alg == "nn_det"
        construct_nn_det!(sol)

    elseif alg == "nn_rand"
        alpha = get(kwargs, :alpha, 0.3)
        construct_nn_rand!(sol; alpha = alpha)

    elseif alg == "nn_rand_multi"
        iters = get(kwargs, :iters, 50)
        alpha = get(kwargs, :alpha, 0.3)
        sol = multistart_randomized_construction(inst; alpha = alpha, iters = iters)

    elseif alg == "pilot"
        construct_pilot!(sol)

    elseif alg == "ls"
        heuristic = GVNS(
            sol,
            [MHMethod("con", construct!)],
            [MHMethod("li1", local_improve!,
                      LocalSearchParams(:inter_route, :first_improvement, false))],
            MHMethod[];
            consider_initial_sol = true,
            titer = titer,
            kwargs...,
        )
        run!(heuristic)
        method_statistics(heuristic.scheduler)
        main_results(heuristic.scheduler)
        copy!(sol, heuristic.scheduler.incumbent)

    elseif alg == "vnd"
        error("Algorithm 'vnd' not yet implemented.")

    elseif alg == "grasp"
        niters = get(kwargs, :niters, 50)   # how many GRASP iterations
        alpha  = get(kwargs, :alpha, 0.3)   # same alpha as in construct_nn_rand!

        # Local search parameters
        ls_par = LocalSearchParams(:two_opt, :first_improvement, false)

        best_sol = nothing
        best_val = Inf

        for it in 1:niters
            s_it = SCFPDPSolution(inst)
            construct_nn_rand!(s_it; alpha = alpha)

            res = Result()
            res.changed = true
            while res.changed
                res.changed = false
                local_improve!(s_it, ls_par, res)
            end

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

    println(sol)
    println("Feasible? ", is_feasible(sol))

    return sol
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
function decompose_objective(s::SCFPDPSolution)
    inst = s.inst

    # route times
    route_times = Float64[]
    total_time = 0.0
    for route in s.routes
        t = route_time(inst, route)
        push!(route_times, t)
        total_time += t
    end

    if isempty(route_times)
        return (Inf, 0.0, Inf)
    end

    fairness = (total_time^2) / (length(route_times) * sum(t^2 for t in route_times))
    obj = total_time + inst.rho * fairness

    return (total_time, fairness, obj)
end
