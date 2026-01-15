include("SCFPDP.jl")

using Random
using Statistics
using StatsBase

export run_genetic!


"""
    initialize_population(inst::SCFPDPInstance, pop_size::Int, construction_method::Symbol)

Initialize population using the specified construction method.
"""
function initialize_population(inst::SCFPDPInstance, pop_size::Int, construction_method::Symbol)
    population = SCFPDPSolution[]
    
    for i in 1:pop_size
        sol = SCFPDPSolution(inst)
        
        if construction_method == :nn_det
            construct_nn_det!(sol)
        elseif construction_method == :nn_rand
            construct_nn_rand!(sol; alpha=0.3)
        else
            error("Unknown construction method: $(construction_method)")
        end
        
        # Ensure objective is calculated
        sol.obj_val = MHLib.calc_objective(sol)
        sol.obj_val_valid = true
        push!(population, sol)
    end
    
    return population
end


"""
    tournament_selection(population::Vector{SCFPDPSolution}, tournament_size::Int) -> SCFPDPSolution

Select a solution from the population using tournament selection.
Returns a reference to the selected solution (not a copy).
"""
function tournament_selection(population::Vector{SCFPDPSolution}, tournament_size::Int)
    # Select random individuals for tournament
    contestants = sample(population, tournament_size, replace=false)
    
    # Return the best one
    return contestants[argmin([MHLib.obj(s) for s in contestants])]
end



# CROSSOVER
"""
    order_crossover!(offspring::SCFPDPSolution, parent1::SCFPDPSolution, parent2::SCFPDPSolution, vehicle_idx::Int)

Perform Order Crossover (OX) on a specific vehicle route between two parents.
Modifies `offspring` in place by combining genetic material from parent1 and parent2 for vehicle `vehicle_idx`.
"""
function order_crossover!(offspring::SCFPDPSolution, parent1::SCFPDPSolution, parent2::SCFPDPSolution, vehicle_idx::Int)
    inst = offspring.inst
    route1 = parent1.routes[vehicle_idx]
    route2 = parent2.routes[vehicle_idx]
    
    # If either route is empty, just copy from parent1
    if isempty(route1) || isempty(route2)
        offspring.routes[vehicle_idx] = copy(route1)
        return
    end
    
    # Extract requests from both routes (not nodes)
    requests1 = Int[]
    for node in route1
        r, is_pickup = node_to_request(inst, node)
        if is_pickup && r != 0 && r ∉ requests1
            push!(requests1, r)
        end
    end
    
    requests2 = Int[]
    for node in route2
        r, is_pickup = node_to_request(inst, node)
        if is_pickup && r != 0 && r ∉ requests2
            push!(requests2, r)
        end
    end
    
    if isempty(requests1) || isempty(requests2)
        offspring.routes[vehicle_idx] = copy(route1)
        return
    end
    
    # Order Crossover: take a segment from parent1, fill remaining with parent2's order
    len1 = length(requests1)
    cut1 = rand(1:len1)
    cut2 = rand(cut1:len1)
    
    # Copy segment from parent1
    offspring_requests = requests1[cut1:cut2]
    
    # Fill remaining positions with requests from parent2 (in order)
    for r in requests2
        if r ∉ offspring_requests
            push!(offspring_requests, r)
        end
    end
    
    # Rebuild route with pickup-dropoff pairs
    empty!(offspring.routes[vehicle_idx])
    for r in offspring_requests
        push!(offspring.routes[vehicle_idx], inst.pickup[r])
        push!(offspring.routes[vehicle_idx], inst.dropoff[r])
    end
end


"""
    route_crossover!(offspring::SCFPDPSolution, parent1::SCFPDPSolution, parent2::SCFPDPSolution)

Perform crossover between parent1 and parent2 to create offspring.
Uses a request-level crossover that preserves served requests from both parents.
"""
function route_crossover!(offspring::SCFPDPSolution, parent1::SCFPDPSolution, parent2::SCFPDPSolution)
    inst = offspring.inst
    
    # Collect all requests served by each parent and their vehicle assignments
    parent1_requests = Dict{Int,Int}()  # request -> vehicle
    parent2_requests = Dict{Int,Int}()  # request -> vehicle
    
    # Cycle through vehicles and their routes
    for k in 1:inst.nk
        for node in parent1.routes[k]
            r, is_pickup = node_to_request(inst, node)
            if is_pickup && r != 0
                parent1_requests[r] = k
            end
        end
        for node in parent2.routes[k]
            r, is_pickup = node_to_request(inst, node)
            if is_pickup && r != 0
                parent2_requests[r] = k
            end
        end
    end
    
    # Combine requests from both parents (union)
    all_requests = union(keys(parent1_requests), keys(parent2_requests))
    
    # Clear offspring routes
    for k in 1:inst.nk
        empty!(offspring.routes[k])
    end
    fill!(offspring.served, false)
    
    # For each request, decide which parent's vehicle assignment to use
    for r in all_requests
        # Randomly choose from which parent to inherit this request's vehicle assignment
        if haskey(parent1_requests, r) && haskey(parent2_requests, r)
            # Both parents have this request, choose randomly
            k = rand() < 0.5 ? parent1_requests[r] : parent2_requests[r]
        elseif haskey(parent1_requests, r)
            # Only parent1 has this request
            k = parent1_requests[r]
        else
            # Only parent2 has this request
            k = parent2_requests[r]
        end
        
        # Add request to the selected vehicle's route
        push!(offspring.routes[k], inst.pickup[r])
        push!(offspring.routes[k], inst.dropoff[r])
        offspring.served[r] = true
    end
    
    # Invalidate objective
    offspring.obj_val_valid = false
end



# MUTATION
"""
    swap_mutation!(sol::SCFPDPSolution)

Perform swap mutation: swap two requests (within same route or between routes).
"""
function swap_mutation!(sol::SCFPDPSolution)
    inst = sol.inst
    
    # Collect all served requests and their vehicle assignments
    served_requests = findall(sol.served)
    isempty(served_requests) && return
    
    # If only one or no requests, cannot swap
    length(served_requests) < 2 && return
    
    # Select two random requests
    r1, r2 = sample(served_requests, 2, replace=false)
    
    # Find which vehicles have these requests
    k1 = findfirst(k -> inst.pickup[r1] ∈ sol.routes[k], 1:inst.nk)
    k2 = findfirst(k -> inst.pickup[r2] ∈ sol.routes[k], 1:inst.nk)
    
    k1 === nothing && return
    k2 === nothing && return
    
    route1 = sol.routes[k1]
    route2 = sol.routes[k2]
    
    # Find positions of pickup and dropoff nodes
    idx_p1 = findfirst(==(inst.pickup[r1]), route1)
    idx_d1 = findfirst(==(inst.dropoff[r1]), route1)
    idx_p2 = findfirst(==(inst.pickup[r2]), route2)
    idx_d2 = findfirst(==(inst.dropoff[r2]), route2)
    
    # Swap the nodes
    route1[idx_p1], route2[idx_p2] = route2[idx_p2], route1[idx_p1]
    route1[idx_d1], route2[idx_d2] = route2[idx_d2], route1[idx_d1]
    
    # Invalidate objective
    sol.obj_val_valid = false
end


"""
    insert_mutation!(sol::SCFPDPSolution)

Perform insert mutation: move a request from one vehicle to another.
"""
function insert_mutation!(sol::SCFPDPSolution)
    inst = sol.inst
    
    served_requests = findall(sol.served)
    isempty(served_requests) && return
    
    # Select random request to move
    r = rand(served_requests)
    
    # Find current vehicle
    k_from = findfirst(k -> inst.pickup[r] ∈ sol.routes[k], 1:inst.nk)
    k_from === nothing && return
    
    # Select different target vehicle
    k_to = rand(filter(k -> k != k_from, 1:inst.nk))
    
    route_from = sol.routes[k_from]
    route_to = sol.routes[k_to]
    
    # Remove from current route
    pickup_node = inst.pickup[r]
    dropoff_node = inst.dropoff[r]
    idx_p = findfirst(==(pickup_node), route_from)
    idx_d = findfirst(==(dropoff_node), route_from)
    
    idx_p === nothing && return
    idx_d === nothing && return
    
    # Delete (higher index first)
    if idx_d > idx_p
        deleteat!(route_from, idx_d)
        deleteat!(route_from, idx_p)
    else
        deleteat!(route_from, idx_p)
        deleteat!(route_from, idx_d)
    end
    
    # Insert at random position in target route
    if isempty(route_to)
        push!(route_to, pickup_node)
        push!(route_to, dropoff_node)
    else
        pos_p = rand(1:length(route_to)+1)
        pos_d = rand(pos_p+1:length(route_to)+2)
        insert!(route_to, pos_p, pickup_node)
        insert!(route_to, pos_d, dropoff_node)
    end
    
    # Invalidate objective
    sol.obj_val_valid = false
end


"""
    add_remove_mutation!(sol::SCFPDPSolution)

Perform add/remove mutation: randomly add an unserved request or remove a served one.
Prioritizes adding requests to reach gamma target.
"""
function add_remove_mutation!(sol::SCFPDPSolution)
    inst = sol.inst
    
    current_served = count(sol.served)
    
    # Prefer adding if below gamma, removing if above
    add_bias = current_served < inst.gamma ? 0.8 : 0.2
    
    if rand() < add_bias
        # Try to add an unserved request
        unserved = findall(.!sol.served)
        if !isempty(unserved)
            r = rand(unserved)
            k = rand(1:inst.nk)
            append_request!(sol, k, r)
        end
    else
        # Try to remove a served request (only if above gamma)
        served = findall(sol.served)
        if !isempty(served) && current_served > inst.gamma
            r = rand(served)
            # Find and remove from route
            for k in 1:inst.nk
                route = sol.routes[k]
                pickup_node = inst.pickup[r]
                dropoff_node = inst.dropoff[r]
                
                idx_p = findfirst(==(pickup_node), route)
                idx_d = findfirst(==(dropoff_node), route)
                
                if idx_p !== nothing && idx_d !== nothing
                    # Delete (higher index first)
                    if idx_d > idx_p
                        deleteat!(route, idx_d)
                        deleteat!(route, idx_p)
                    else
                        deleteat!(route, idx_p)
                        deleteat!(route, idx_d)
                    end
                    sol.served[r] = false
                    break
                end
            end
        end
    end
    
    # Invalidate objective
    sol.obj_val_valid = false
end



# REPAIR
"""
    repair_solution!(sol::SCFPDPSolution)

Repair a solution by ensuring it satisfies both feasibility constraints and the gamma requirement.
Iteratively removes or adds requests until the solution is feasible with the correct number of served requests.
"""
function repair_solution!(sol::SCFPDPSolution)
    inst = sol.inst

    # Instead of doing repair, we could also just construct a new,
    # randomized solution from scratch, and replace the current child with it.
            # new_sol = SCFPDPSolution(inst)
            # construct_nn_rand!(new_sol; alpha=0.3)
            # copy!(sol, new_sol)


    # But, we are going the repair route here:

    # Unified repair loop: handle both infeasibility and gamma violations
    while !is_feasible_with_gamma_check(sol)
        current_served = count(sol.served)
        
        # Case 1: Solution violates capacity/precedence constraints OR has too many requests
        if !is_feasible(sol) || current_served > inst.gamma
            # Remove a random served request
            served = findall(sol.served)
            if isempty(served)
                break  # No requests to remove
            end
            
            r = rand(served)
            # Remove this request from its route
            for k in 1:inst.nk
                route = sol.routes[k]
                pickup_node = inst.pickup[r]
                dropoff_node = inst.dropoff[r]
                
                idx_p = findfirst(==(pickup_node), route)
                idx_d = findfirst(==(dropoff_node), route)
                
                if idx_p !== nothing && idx_d !== nothing
                    if idx_d > idx_p
                        deleteat!(route, idx_d)
                        deleteat!(route, idx_p)
                    else
                        deleteat!(route, idx_p)
                        deleteat!(route, idx_d)
                    end
                    sol.served[r] = false
                    break
                end
            end
            
        # Case 2: Solution is feasible but has too few requests
        elseif current_served < inst.gamma
            # Try to add an unserved request
            unserved = findall(.!sol.served)
            if isempty(unserved)
                break  # No requests to add
            end
            
            shuffle!(unserved)
            added = false
            
            # Try to add one request to any vehicle
            for r in unserved
                for k in 1:inst.nk
                    temp_sol = copy(sol)
                    append_request!(temp_sol, k, r)
                    
                    if is_feasible(temp_sol)
                        # Accept this addition
                        append_request!(sol, k, r)
                        added = true
                        break
                    end
                end
                if added
                    break
                end
            end
            
            # If we couldn't add any request, we're stuck
            if !added
                break
            end
        end
    end
    
    sol.obj_val_valid = false
end


# """
#     greedy_completion!(sol::SCFPDPSolution)

# Greedily add unserved requests to the solution until gamma is reached or no more feasible additions.
# Tries to add requests to the vehicle that results in the smallest objective increase.
# """
# function greedy_completion!(sol::SCFPDPSolution)
#     inst = sol.inst
    
#     # Keep trying to add requests until we reach gamma or can't add more
#     while count(sol.served) < inst.gamma
#         unserved = findall(.!sol.served)
#         if isempty(unserved)
#             break
#         end
        
#         best_addition = nothing
#         best_cost = Inf
        
#         # Try adding each unserved request to each vehicle
#         for r in unserved
#             for k in 1:inst.nk
#                 temp_sol = copy(sol)
#                 append_request!(temp_sol, k, r)
                
#                 if is_feasible(temp_sol)
#                     cost = MHLib.calc_objective(temp_sol)
#                     if cost < best_cost
#                         best_cost = cost
#                         best_addition = (r, k)
#                     end
#                 end
#             end
#         end
        
#         # If we found a feasible addition, apply it
#         if best_addition !== nothing
#             r, k = best_addition
#             append_request!(sol, k, r)
#         else
#             # No more feasible additions
#             break
#         end
#     end
    
#     sol.obj_val_valid = false
# end


"""
    run_genetic!(sol::SCFPDPSolution;
                 pop_size::Int = 50,
                 crossover_rate::Float64 = 0.9,
                 mutation_rate::Float64 = 0.2,
                 tournament_size::Int = 3,
                 elite_size::Int = 2,
                 ttime::Float64 = 10.0,
                 max_generations::Int = 1000,
                 construction_method::Symbol = :nn_rand,
                 seed::Union{Nothing,Int} = nothing,
                 verbose::Bool = false)

Run a genetic algorithm to solve the SCFPDP.

This function implements a genetic/evolutionary algorithm and returns results
in a format consistent with other metaheuristics (like ACO).

# Parameters
- `sol`: Initial solution (will be overwritten with best found solution)
- `pop_size`: Population size
- `crossover_rate`: Probability of crossover
- `mutation_rate`: Probability of mutation
- `tournament_size`: Size of tournament for selection
- `elite_size`: Number of elite individuals to preserve
- `ttime`: Maximum runtime in seconds
- `max_generations`: Maximum number of generations
- `construction_method`: Method for population initialization (:nn_det, :nn_rand)
- `seed`: Random seed for reproducibility
- `verbose`: Print progress information

# Returns
- `sol`: Best solution found (modified in-place)
- `stats`: Named tuple with (iters=generations, best_val=objective, time=runtime)
"""
function run_genetic!(sol::SCFPDPSolution;
                     pop_size::Int = 50,
                     crossover_rate::Float64 = 0.9,
                     mutation_rate::Float64 = 0.2,
                     tournament_size::Int = 3,
                     elite_size::Int = 2,
                     ttime::Float64 = 10.0,
                     max_generations::Int = 1000,
                     construction_method::Symbol = :nn_rand,
                     seed::Union{Nothing,Int} = nothing,
                     verbose::Bool = false)

    inst = sol.inst

    # Set random seed if provided
    if seed !== nothing
        Random.seed!(seed)
    end
    
    if verbose
        println("\n" * "="^60)
        println("GENETIC ALGORITHM FOR SCFPDP")
        println("="^60)
        println("Population size: $pop_size")
        println("Crossover rate: $crossover_rate")
        println("Mutation rate: $mutation_rate")
        println("Tournament size: $tournament_size")
        println("Elite size: $elite_size")
        println("Max generations: $max_generations")
        println("Time limit: $(ttime)s")
        println("Construction method: $construction_method")
        println("="^60 * "\n")
    end
    
    start_time = time()
    
    # Initialize population
    if verbose
        println("Initializing population...")
    end
    
    population = initialize_population(inst, pop_size, construction_method)
    
    # Track best solution
    best_sol = copy(population[argmin([MHLib.obj(s) for s in population])])
    best_obj = MHLib.obj(best_sol)
    
    # Keep track of generations and generations without improvement (for early stopping)
    generation = 0
    generation_no_improvement = 0
    
    if verbose
        println("Initial best objective: $(round(best_obj, digits=2))")
        println()
    end


    while (time() - start_time) < ttime && generation < max_generations
        generation += 1
        
        # Sort population by fitness
        sort!(population, by=s -> MHLib.obj(s))
        
        # Create offspring population
        offspring = SCFPDPSolution[]
        
        # Elitism: keep best solutions for next generation;
        # these don't undergo any selection/crossover/mutation
        for i in 1:elite_size
            push!(offspring, copy(population[i]))
        end
        
        # Generate rest of offspring
        while length(offspring) < pop_size
            # Check time limit during generation
            if (time() - start_time) >= ttime
                break
            end
            
            # Selection
            parent1 = tournament_selection(population, tournament_size)
            parent2 = tournament_selection(population, tournament_size)
            

            child = copy(parent1)
            
            # Crossover
            if rand() < crossover_rate
                route_crossover!(child, parent1, parent2)
            end
            
            # Mutation
            if rand() < mutation_rate
                mutation_type = rand()
                if mutation_type < 0.5
                    swap_mutation!(child)
                elseif mutation_type < 0.9
                    insert_mutation!(child)
                else
                    add_remove_mutation!(child)
                end
            end
            
            # Repair
            if !is_feasible(child)
                repair_solution!(child)
            end

            # If we haven't reached gamma yet, try to complete greedily
            ### This is now being handled inside repair_solution!
            # if count(child.served) < inst.gamma
            #     greedy_completion!(child)
            # end
            

            child.obj_val = MHLib.calc_objective(child)
            child.obj_val_valid = true
            
            push!(offspring, child)
        end
        
        # Replace population for next generation
        population = offspring
        
        # Update best solution
        current_best = population[argmin([MHLib.obj(s) for s in population])]
        current_best_obj = MHLib.obj(current_best)
        
        if current_best_obj < best_obj
            best_obj = current_best_obj
            best_sol = copy(current_best)
            generation_no_improvement = 0
            if verbose
                println("Generation $generation: New best = $(round(best_obj, digits=2))")
            end
        else
            generation_no_improvement += 1
        end
        
        # Progress reporting
        if verbose && generation % 10 == 0
            avg_obj = mean([MHLib.obj(s) for s in population])
            elapsed = round(time() - start_time, digits=2)
            println("Generation $generation ($(elapsed)s): Best = $(round(best_obj, digits=2)), Avg = $(round(avg_obj, digits=2))")
        end
        
        # Stop early if there's no improvement for a while
        if generation_no_improvement > 50
            if verbose
                println("\nEarly stopping: No improvement for 50 generations")
            end
            break
        end
    end
    
    elapsed_time = time() - start_time
    
    if verbose
        println("\n" * "="^60)
        println("GENETIC ALGORITHM COMPLETED")
        println("="^60)
        println("Generations: $generation")
        println("Best objective: $(round(best_obj, digits=2))")
        println("Total time: $(round(elapsed_time, digits=2))s")
        println("Feasible: $(is_feasible(best_sol))")
        println("Requests served: $(count(best_sol.served)) / $(inst.gamma) / $(inst.n)")
        println("="^60)
    end
    
    # Copy best solution back to input solution (overwrite)
    copy!(sol, best_sol)
    
    return sol, (iters=generation, best_val=best_obj, time=elapsed_time)
end