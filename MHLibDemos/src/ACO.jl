include("SCFPDP.jl")

using Random
using StatsBase

# pheromones : initial pheromone level
function init_tau(inst::SCFPDPInstance; tau0::Float64 = 1.0)
    return fill(tau0, inst.n, inst.nk)
end

# previous evaporation
function evaporate!(tau::Matrix{Float64}; rho::Float64 = 0.1)
    tau .*= (1.0 - rho)
    return tau
end

# candidate list : ants create a solution : here what ant is allowed to do next
# N_i^k feasible (request r, vehicle k) insertions
function aco_candidates(s::SCFPDPSolution, remaining::Vector{Int})
    inst = s.inst
    base_obj = MHLib.calc_objective(s) 
    candidates = Tuple{Float64,Int,Int}[]

    for r in remaining
        for k in 1:inst.nk
            tmp = copy(s)
            append_request!(tmp, k, r)
            is_feasible(tmp) || continue

            # local informaiton: heuristics step 1 page 106 lecture notes
            new_obj = MHLib.calc_objective(tmp)
            Δ = new_obj - base_obj
            push!(candidates, (Δ, r, k)) # candidate evaluation for later probabilitic selection
        end
    end
    return candidates
end

# local information η is defined analogously to TSP visibility from lecture notes: 
# inverse objective increase caused by inserting a request into a vehicle route + small eps to avoid division by zero


# helper: normalize a probability vector in-place ∑​τilα​⋅ηilβ​ (denominator of probability formula)
function normalize_probs!(p::Vector{Float64})
    s = sum(p)
    if !(s > 0.0) || !isfinite(s)
        fill!(p, 1.0 / length(p))
    else
        for i in eachindex(p)
            p[i] /= s
        end
    end
    return p
end

# function aco_probs(candidates, tau::Matrix{Float64};
#                    alpha::Float64 = 1.0,
#                    beta::Float64  = 3.0,
#                    eps::Float64   = 1e-9)

#     p = Float64[]
#     for (Δ, r, k) in candidates
#         # η = 1.0 / (Δ + eps)
#         # η = 1.0 / (max(Δ, 0.0) + eps)
#         Δpos = max(Δ, 0.0)
#         η = 1.0 / (Δpos + eps)

#         # combine pheromone τ and local information η with parameters alpha and beta τijα​⋅ηijβ​
#         score = (tau[r, k]^alpha) * (η^beta)
#         push!(p, score)
#     end
#     return normalize_probs!(p)
# end

# function aco_probs(candidates, tau::Matrix{Float64};
#                    alpha::Float64 = 1.0,
#                    beta::Float64  = 3.0,
#                    eps::Float64   = 1e-9)

#     p = Float64[]
#     for (Δ, r, k) in candidates
#         Δpos = max(Δ, 0.0)          # we dont get negative increases
#         η = 1.0 / (Δpos + eps)
#         score = (tau[r, k]^alpha) * (η^beta)
#         push!(p, score)
#     end
#     return normalize_probs!(p)
# end
function aco_probs(candidates, tau::Matrix{Float64};
                   alpha::Float64 = 1.0,
                   beta::Float64  = 3.0,
                   eps::Float64   = 1e-9)

    # shift deltas so the best delta becomes 0
    Δmin = minimum(c[1] for c in candidates)

    p = Float64[]
    for (Δ, r, k) in candidates
        # shifted delta is always >= 0
        Δshift = (Δ - Δmin)
        η = 1.0 / (Δshift + eps)

        score = (tau[r, k]^alpha) * (η^beta)
        push!(p, score)
    end

    return normalize_probs!(p)
end




# pheromone deposit page 109
function deposit_pheromones!(tau::Matrix{Float64},
                             sols;         
                             Q::Float64 = 1.0,
                             eps::Float64 = 1e-9)

    for (components, obj_val) in sols
        Δτ = Q / (obj_val + eps)

        for (r, k) in components
            tau[r, k] += Δτ
        end
    end

    return tau
end


# Deposit pheromones only from the best solution of the iteration (elitist update).
function deposit_pheromones_best!(tau::Matrix{Float64},
                                  sols;
                                  Q::Float64 = 0.2,
                                  eps::Float64 = 1e-9)

    isempty(sols) && return tau

    best_idx = 1
    best_val = sols[1][2]
    @inbounds for i in 2:length(sols)
        v = sols[i][2]
        if v < best_val
            best_val = v
            best_idx = i
        end
    end

    components, obj_val = sols[best_idx]
    Δτ = Q / (obj_val + eps)

    @inbounds for (r, k) in components
        tau[r, k] += Δτ
    end

    return tau
end


# evaporate + deposit pheromones -> τ(t+1)=(1−ρ)τ(t)+Δτ(t) 

# Procedure AS

function sample_candidate(candidates, probs)
    idx = sample(1:length(candidates), Weights(probs))
    return candidates[idx]  # (Δ, r, k)
end

function apply_candidate!(s::SCFPDPSolution, r::Int, k::Int)
    append_request!(s, k, r)
    return s
end


# Local improvement helpers
"""
Extract (r,k) assignments from the FINAL solution (after local search).
We record each served request once, based on pickup nodes present in route k.
"""
function extract_components(s::SCFPDPSolution)
    inst = s.inst
    comps = Tuple{Int,Int}[]
    for k in 1:inst.nk
        for node in s.routes[k]
            r, is_pickup = node_to_request(inst, node)
            if is_pickup && r != 0
                push!(comps, (r, k))
            end
        end
    end
    return comps
end

"""
VND-style local improvement (first-improvement).
Uses use_delta=false as this is now implemented in our project.
"""
function local_improve_vnd!(s::SCFPDPSolution;
                            strategy::Symbol = :first_improvement,
                            use_delta::Bool = false,
                            max_rounds::Int = 10)
    res = MHLib.Result()
    for _ in 1:max_rounds
        changed = false
        changed |= MHLib.local_improve!(s, LocalSearchParams(neighborhood=:relocate,    strategy=strategy, use_delta=use_delta), res)
        changed |= MHLib.local_improve!(s, LocalSearchParams(neighborhood=:inter_route, strategy=strategy, use_delta=use_delta), res)
        changed |= MHLib.local_improve!(s, LocalSearchParams(neighborhood=:two_opt,     strategy=strategy, use_delta=use_delta), res)
        changed || break
    end
    s.obj_val = MHLib.calc_objective(s)
    s.obj_val_valid = true
    return s
end


function construct_aco_ant!(s::SCFPDPSolution, tau::Matrix{Float64};
                           alpha::Float64 = 1.0,
                           beta::Float64  = 3.0,
                           eps::Float64   = 1e-9)

    inst = s.inst
    initialize!(s)

    remaining = collect(1:inst.n)
    components = Tuple{Int,Int}[] 

    while count(s.served) < inst.gamma && !isempty(remaining)
        candidates = aco_candidates(s, remaining)
        isempty(candidates) && break

        probs = aco_probs(candidates, tau; alpha=alpha, beta=beta, eps=eps)
        (Δ, r, k) = sample_candidate(candidates, probs)

        apply_candidate!(s, r, k)
        push!(components, (r, k))

        pos = findfirst(==(r), remaining)
        pos === nothing || deleteat!(remaining, pos)
    end

    s.obj_val = MHLib.calc_objective(s)
    s.obj_val_valid = true

    return components
end


# function run_aco!(sol::SCFPDPSolution;
#                   num_ants::Int = 30,
#                   alpha::Float64 = 1.0,
#                   beta::Float64 = 3.0,
#                   rho::Float64 = 0.1,
#                   ttime::Float64 = 10.0,
#                   seed::Int = 1,
#                   tau0::Float64 = 1.0,
#                   Q::Float64 = 1.0,
#                   eps::Float64 = 1e-9)

#     inst = sol.inst
#     Random.seed!(seed)

#     # initialize pheromones
#     tau = init_tau(inst; tau0=tau0)

#     # keep best solution found
#     best_val = Inf
#     best_sol = nothing

#     start = time()
#     iters = 0

#     while (time() - start) < ttime
#         sols = Tuple{Vector{Tuple{Int,Int}}, Float64}[]

#         # for each ant: construct a solution
#         for _ in 1:num_ants
#             s = SCFPDPSolution(inst)
#             comps = construct_aco_ant!(s, tau; alpha=alpha, beta=beta, eps=eps)

#             push!(sols, (comps, s.obj_val))

#             if s.obj_val < best_val
#                 best_val = s.obj_val
#                 best_sol = copy(s)
#             end
#         end

#         # pheromone update: evaporate + deposit
#         evaporate!(tau; rho=rho)
#         deposit_pheromones!(tau, sols; Q=Q, eps=eps)

#         iters += 1
#     end

#     best_sol === nothing && error("ACO failed to build any solution.")

#     # write best back into the provided 'sol'
#     copy!(sol, best_sol)

#     return sol, (iters=iters, best_val=best_val, time=time()-start)
# end



# function run_aco!(sol::SCFPDPSolution;
#                   num_ants::Int = 30,
#                   alpha::Float64 = 1.0,
#                   beta::Float64 = 3.0,
#                   rho::Float64 = 0.1,
#                   ttime::Float64 = 10.0,
#                   seed::Int = 1,
#                   tau0::Float64 = 1.0,
#                   Q::Float64 = 1.0,
#                   eps::Float64 = 1e-9,
#                   do_local_improve::Bool = true,
#                   improve_only_best_ant::Bool = true)

#     inst = sol.inst
#     Random.seed!(seed)

#     tau = init_tau(inst; tau0=tau0)

#     best_val = Inf
#     best_sol = nothing

#     start = time()
#     iters = 0

#     while (time() - start) < ttime
#         sols = Tuple{Vector{Tuple{Int,Int}}, Float64}[]
#         iter_best_s = nothing
#         iter_best_val = Inf

#         ants = Vector{SCFPDPSolution}(undef, num_ants)

#         # 1) Construct solutions
#         for a in 1:num_ants
#             s = SCFPDPSolution(inst)
#             construct_aco_ant!(s, tau; alpha=alpha, beta=beta, eps=eps)
#             ants[a] = s

#             if s.obj_val < iter_best_val
#                 iter_best_val = s.obj_val
#                 iter_best_s = s
#             end
#         end

#         # 2) Optional local improvement
#         if do_local_improve
#             if improve_only_best_ant
#                 # improve just the iteration-best 
#                 local_improve_vnd!(iter_best_s; strategy=:first_improvement, use_delta=false, max_rounds=10)
#             else
#                 # improve all ants 
#                 for s in ants
#                     local_improve_vnd!(s; strategy=:first_improvement, use_delta=false, max_rounds=10)
#                 end
#             end
#         end

#         # 3) Collect deposit info 
#         for s in ants
#             comps = extract_components(s)
#             push!(sols, (comps, s.obj_val))

#             if s.obj_val < best_val
#                 best_val = s.obj_val
#                 best_sol = copy(s)
#             end
#         end

#         # 4) Update pheromones
#         evaporate!(tau; rho=rho)
#         # deposit_pheromones!(tau, sols; Q=Q, eps=eps)
#         deposit_pheromones_best!(tau, sols; Q=Q, eps=eps)


#         iters += 1
#     end

#     best_sol === nothing && error("ACO failed to build any solution.")

#     copy!(sol, best_sol)
#     return sol, (iters=iters, best_val=best_val, time=time()-start)
# end


function run_aco!(sol::SCFPDPSolution;
                  num_ants::Int = 30,
                  alpha::Float64 = 1.0,
                  beta::Float64 = 3.0,
                  rho::Float64 = 0.1,
                  ttime::Float64 = 10.0,
                  seed::Int = 1,
                  tau0::Float64 = 1.0,
                  Q::Float64 = 1.0,
                  eps::Float64 = 1e-9,
                  lsparams::Union{Nothing,LocalSearchParams} = nothing,
                  ls_iters::Int = 0,
                  # pheromone bounds needed? 
                  tau_min::Float64 = 1e-4,
                  tau_max::Float64 = 10.0)

    inst = sol.inst
    Random.seed!(seed)

    tau = init_tau(inst; tau0=tau0)

    best_val = Inf
    best_sol = nothing

    start = time()
    iters = 0

    while (time() - start) < ttime
        sols = Tuple{Vector{Tuple{Int,Int}}, Float64}[]

        for _ in 1:num_ants
            s = SCFPDPSolution(inst)

            # construct one ant
            construct_aco_ant!(s, tau; alpha=alpha, beta=beta, eps=eps)

            # optional local search
            # if lsparams !== nothing && ls_iters > 0
            #     r = MHLib.Result()
            #     for _ls in 1:ls_iters
            #         MHLib.local_improve!(s, lsparams, r)
            #         r.changed || break
            #         r.changed = false
            #     end
            #     s.obj_val = MHLib.calc_objective(s)
            #     s.obj_val_valid = true
            # end
            # 2) optional local search (VND: relocate + inter_route + two_opt)
            if lsparams !== nothing && ls_iters > 0
                # interpret ls_iters as number of VND rounds
                local_improve_vnd!(
                    s;
                    strategy = :first_improvement,
                    use_delta = false,
                    max_rounds = ls_iters
                )
            end


            # IMPORTANT deposit components after local search
            final_comps = extract_components(s)
            push!(sols, (final_comps, s.obj_val))

            if s.obj_val < best_val
                best_val = s.obj_val
                best_sol = copy(s)
            end
        end

        evaporate!(tau; rho=rho)
        deposit_pheromones_best!(tau, sols; Q=Q, eps=eps)

        @inbounds for i in eachindex(tau)
            if tau[i] < tau_min
                tau[i] = tau_min
            elseif tau[i] > tau_max
                tau[i] = tau_max
            end
        end

        iters += 1
    end

    best_sol === nothing && error("ACO failed to build any solution.")
    copy!(sol, best_sol)

    return sol, (iters=iters, best_val=best_val, time=time()-start)
end
