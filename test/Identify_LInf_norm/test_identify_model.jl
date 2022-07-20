module TestMain

using Test
using LinearAlgebra
using StaticArrays
using JuMP
using GLPK
@static if isdefined(Main, :TestLocal)
    include("../../src/HybridSystemIdentification.jl")
else
    using HybridSystemIdentification
end
HSI = HybridSystemIdentification

sleep(0.1) # used for good printing
println("Started test")

solver = optimizer_with_attributes(GLPK.Optimizer, "msg_lev"=>GLPK.GLP_MSG_ON)
# solver = optimizer_with_attributes(Gurobi.Optimizer)

#===============================================================================
DATA
===============================================================================#
α = 0.1*(2*π)
β = 0.5
γ = 0.8
A1 = @SMatrix [β*cos(α) -β*sin(α); β*sin(α) β*cos(α)]
A2 = @SMatrix [γ 1.0; 0.0 γ]
B1 = @SMatrix [1.0; 0.0]
B2 = @SMatrix [0.0; 1.0]
A_list = (A1, A2)
B_list = (B1, B2)

n_obs = 100
noise = 0.01
x0 = @SVector [0.1, -0.1]
q_seq = Vector{Int}(undef, n_obs)
u_seq = [SVector(1.0) for i = 1:n_obs]
x1_seq = Vector{SVector{2,Float64}}(undef, n_obs)
y1_seq = Vector{SVector{2,Float64}}(undef, n_obs)
x2_seq = Vector{SVector{2,Float64}}(undef, n_obs)
y2_seq = Vector{SVector{2,Float64}}(undef, n_obs)
y_seq = Vector{SVector{2,Float64}}(undef, n_obs + 1)
x = x0

fnoise(i) = mod(i, 3) == 0 ? noise : -noise

for i = 1:n_obs
    q = mod(i, 4) == 0 ? 1 : 2
    q_seq[i] = q
    A = A_list[q_seq[i]]; B = B_list[q_seq[i]]
    x1_seq[i] = x
    y1_seq[i] = x1_seq[i] .+ fnoise(i*i + 1)
    x2_seq[i] = A*x + B*u_seq[i]
    y2_seq[i] = x2_seq[i] .+ fnoise(i*i*i + 1)
    global x = A*x + B*u_seq[i]
    y_seq[i] = y1_seq[i]
end

y_seq[n_obs + 1] = y2_seq[end]

#===============================================================================
TESTS
===============================================================================#
@testset "identify A" begin
    prob = HSI.IdentifyProblemX(2, 1, 2, n_obs, u_seq, x1_seq, x2_seq)
    μA_list = [zeros(SMatrix{2,2}) for q = 1:prob.n_mode]
    δA_list = [10.0 for q = 1:prob.n_mode]
    Ao_list, ropt, flag = HSI.identify_model(
        prob, Val(HSI.L∞Norm), Val(HSI.BKnown),
        q_seq, μA_list, δA_list, B_list, solver)
    @test flag
    @test ropt < 1e-5
    @test maximum(i -> norm(Ao_list[i] - A_list[i], Inf), 1:2) < 1e-5
    prob = HSI.IdentifyProblemX(2, 1, 2, n_obs, u_seq, y1_seq, y2_seq)
    Ao_list, ropt, flag = HSI.identify_model(
        prob, Val(HSI.L∞Norm), Val(HSI.BKnown),
        q_seq, μA_list, δA_list, B_list, solver)
    @test flag
    @test ropt < 0.03
    @test maximum(i -> norm(Ao_list[i] - A_list[i], Inf), 1:2) < 0.015
end

@testset "identify B" begin
    prob = HSI.IdentifyProblemX(2, 1, 2, n_obs, u_seq, x1_seq, x2_seq)
    μB_list = [zeros(SMatrix{2,1}) for q = 1:prob.n_mode]
    δB_list = [10.0 for q = 1:prob.n_mode]
    Bo_list, ropt, flag = HSI.identify_model(
        prob, Val(HSI.L∞Norm), Val(HSI.AKnown),
        q_seq, A_list, μB_list, δB_list, solver)
    @test flag
    @test ropt < 1e-5
    @test maximum(i -> norm(Bo_list[i] - B_list[i], Inf), 1:2) < 1e-5
    prob = HSI.IdentifyProblemX(2, 1, 2, n_obs, u_seq, y1_seq, y2_seq)
    Bo_list, ropt, flag = HSI.identify_model(
        prob, Val(HSI.L∞Norm), Val(HSI.AKnown),
        q_seq, A_list, μB_list, δB_list, solver)
    @test flag
    @test ropt < 0.02
    @test maximum(i -> norm(Bo_list[i] - B_list[i], Inf), 1:2) < 0.02
end

@testset "identify A B" begin
    prob = HSI.IdentifyProblemX(2, 1, 2, n_obs, u_seq, x1_seq, x2_seq)
    μA_list = [zeros(SMatrix{2,2}) for q = 1:prob.n_mode]
    δA_list = [10.0 for q = 1:prob.n_mode]
    μB_list = [zeros(SMatrix{2,1}) for q = 1:prob.n_mode]
    δB_list = [10.0 for q = 1:prob.n_mode]
    Ao_list, Bo_list, ropt, flag = HSI.identify_model(
        prob, Val(HSI.L∞Norm), nothing,
        q_seq, μA_list, δA_list, μB_list, δB_list, solver)
    @test flag
    @test ropt < 1e-5
    @test maximum(i -> norm(Ao_list[i] - A_list[i], Inf), 1:2) < 1e-5
    @test maximum(i -> norm(Bo_list[i] - B_list[i], Inf), 1:2) < 1e-5
    prob = HSI.IdentifyProblemX(2, 1, 2, n_obs, u_seq, y1_seq, y2_seq)
    Ao_list, Bo_list, ropt, flag = HSI.identify_model(
        prob, Val(HSI.L∞Norm), nothing,
        q_seq, μA_list, δA_list, μB_list, δB_list, solver)
    @test flag
    @test ropt < 0.02
    @test maximum(i -> norm(Ao_list[i] - A_list[i], Inf), 1:2) < 1e-5
    @test maximum(i -> norm(Bo_list[i] - B_list[i], Inf), 1:2) < 0.02
end

sleep(0.1) # used for good printing
println("End test")

end  # module TestMain