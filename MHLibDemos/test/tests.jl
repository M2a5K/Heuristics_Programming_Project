# tests.jl
#
# Unit tests for MHLibDemos.
#
# Always performed in the test directory within the test environment.

using TestItems

# This initialization is executed before each test item
@testsnippet MHLibTestInit begin
    using Random
    using MHLib
    using MHLibDemos
    Random.seed!(1)
    mhroot = dirname(dirname(pathof(MHLibDemos)))
    datapath = joinpath(mhroot, "instances", "50", "train")
end

@testitem "SCFPDPInit" setup=[MHLibTestInit] begin
    inst = SCFPDPInstance(joinpath(datapath, "instance1_nreq50_nveh2_gamma50.txt"))

    @test inst.n == 50
    @test inst.nk == 2
    @test inst.C == 100
    @test inst.gamma == 50
    @test inst.rho == 40.08

    @test length(inst.c) == 50
    @test inst.c == [20, 22, 30, 38, 34, 49, 25, 50, 19, 18, 20, 18, 27, 39, 20, 31, 25, 31, 39, 32,
        36, 35, 21, 22, 31, 29, 29, 32, 26, 34, 17, 21, 18, 25, 26, 33, 24, 26,
        26, 40, 32, 30, 24, 32, 28, 19, 27, 43, 40, 17]

    @test inst.depot == 1
    @test inst.pickup == collect(2:51)
    @test inst.dropoff == collect(52:101)

    @test length(inst.coords) == 101
    @test inst.coords == Tuple{Int,Int}[
        (35, 35), (61, 9), (43, 25), (45, 28), (70, 31), (57, 45), (18, 2), (59, 34), (48, 30), (9, 67),
        (6, 66), (28, 10), (20, 11), (1, 7), (39, 18), (38, 50), (7, 60), (64, 2), (56, 61), (5, 11),
        (48, 16), (27, 49), (6, 62), (37, 23), (64, 39), (19, 59), (15, 29), (45, 68), (7, 36), (16, 39),
        (49, 13), (47, 41), (6, 64), (48, 23), (53, 49), (11, 3), (59, 34), (19, 2), (59, 13), (59, 24),
        (69, 64), (18, 37), (40, 11), (60, 1), (29, 67), (18, 70), (61, 13), (60, 14), (27, 58), (49, 45),
        (57, 67), (63, 61), (3, 71), (35, 37), (27, 46), (21, 17), (37, 18), (55, 56), (34, 46), (17, 21),
        (16, 47), (38, 69), (39, 9), (57, 69), (56, 49), (6, 45), (69, 53), (6, 50), (32, 57), (20, 59),
        (55, 8), (68, 30), (15, 58), (57, 55), (4, 29), (18, 31), (65, 39), (15, 12), (5, 4), (57, 42),
        (40, 15), (9, 37), (17, 1), (67, 4), (34, 67), (69, 24), (62, 47), (49, 6), (24, 17), (51, 54),
        (68, 38), (8, 56), (14, 0), (15, 70), (63, 65), (26, 44), (61, 61), (19, 5), (61, 67), (4, 15),
        (71, 23)]

    @test length(inst.d) == 10201
    @test all(inst.d[i,i] == 0.0 for i in 1:size(inst.d, 1))
end

@testitem "SCFPDPCalcObjective" setup=[MHLibTestInit] begin
    inst = SCFPDPInstance(joinpath(datapath, "instance1_nreq50_nveh2_gamma50.txt"))
    sol = SCFPDPSolution(inst)

    initialize!(sol)
    @test !sol.obj_val_valid
    obj_value = obj(sol)
    @test obj_value >= 0
    @test sol.obj_val_valid  # calling MHLib.jl's obj method sets obj_val_valid to true

    println(sol)
    println("Objective value after calling initialize!: $obj_value")
end

@testitem "SCF_PDP_NNDeterministic" setup=[MHLibTestInit] begin
    inst = SCF_PDP_Instance(joinpath(datapath, "instance1_nreq50_nveh2_gamma50.txt"))
    sol  = SCF_PDP_Solution(inst)

    MHLibDemos.construct_nn_det!(sol)
    @test MHLibDemos.is_feasible(sol)
    @test isfinite(sol.obj_val)
    @test count(sol.served) <= inst.gamma
    @test count(sol.served) > 0
    served_idx = findall(sol.served)
    nodes = vcat(sol.routes...)  
    for r in served_idx
        p = inst.pickup[r]
        q = inst.dropoff[r]
        @test p in nodes
        @test q in nodes
    end
    @info "NN_det produced" routes=sol.routes served=count(sol.served) obj=sol.obj_val
end


@testitem "SCF_PDP_NNRand" setup=[MHLibTestInit] begin
    inst = SCF_PDP_Instance(joinpath(datapath, "instance1_nreq50_nveh2_gamma50.txt"))
    sol  = SCF_PDP_Solution(inst)

    MHLibDemos.construct_nn_rand!(sol; alpha=0.3)
    @test MHLibDemos.is_feasible(sol)
    @test isfinite(sol.obj_val)
    @test count(sol.served) <= inst.gamma
    @test count(sol.served) > 0
end

@testitem "SCF_PDP_NNRand_MultiStart" setup=[MHLibTestInit] begin
    inst = SCF_PDP_Instance(joinpath(datapath, "instance1_nreq50_nveh2_gamma50.txt"))

    # local redefinition to be able to run the test 
    function multistart_randomized_construction_test(inst; alpha=0.3, iters=20)
        best = nothing
        best_val = Inf

        for _ in 1:iters
            s = SCF_PDP_Solution(inst)
            MHLibDemos.construct_nn_rand!(s; alpha)

            if s.obj_val < best_val
                best = copy(s)
                best_val = s.obj_val
            end
        end

        return best
    end

    sol_best = multistart_randomized_construction_test(inst; alpha=0.3, iters=20)

    @test MHLibDemos.is_feasible(sol_best)
    @test isfinite(sol_best.obj_val)
    @test count(sol_best.served) == inst.gamma
end


@testitem "SCF_PDP_Pilot" setup=[MHLibTestInit] begin
    inst = SCF_PDP_Instance(joinpath(datapath, "instance1_nreq50_nveh2_gamma50.txt"))
    sol  = SCF_PDP_Solution(inst)

    MHLibDemos.construct_pilot!(sol)
    @test MHLibDemos.is_feasible(sol)
    @test isfinite(sol.obj_val)
    @test count(sol.served) <= inst.gamma
    @test count(sol.served) > 0
end
