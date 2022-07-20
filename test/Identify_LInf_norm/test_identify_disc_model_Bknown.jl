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
VLIN = Val(HSI.L∞Norm)
VPId = Val(HSI.PriorIdentical)

sleep(0.1) # used for good printing
println("Started test")

solver = optimizer_with_attributes(GLPK.Optimizer, "msg_lev"=>GLP_MSG_ON)
# solver = optimizer_with_attributes(Gurobi.Optimizer)

#===============================================================================
DATA
===============================================================================#
α = 0.1*(2*π)
β = 0.5
γ = 0.8
A1 = @SMatrix [β*cos(α) -β*sin(α); β*sin(α) β*cos(α)]
A2 = @SMatrix [γ 1.0; 0.0 γ]
B = @SMatrix [0.0; 1.0]
A_list = (A1, A2)

n_obs = 70
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
    q = mod(i - 1, 4) == 0 ? 1 : 2
    q_seq[i] = q
    A = A_list[q_seq[i]]
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
@testset "identify disc A Identical Noiseless" begin
    prob = HSI.IdentifyProblemX(2, 1, 2, n_obs, u_seq, x1_seq, x2_seq)
    μA = zeros(SMatrix{2,2})
    δA = 10.0
    ϵ = 1e-6
    qo_seq, Ao_list, io_coll, flag = HSI.identify_disc_model(
        prob, VLIN, Val(HSI.BKnown), VPId,
        μA, δA, B, ϵ, solver, Dbin=10)
    qo_seq = 3 .- qo_seq
    Ao_list[1], Ao_list[2] = Ao_list[2], Ao_list[1]
    @test flag
    @test isempty(io_coll)
    @test norm(qo_seq - q_seq) < 1e-14
    @test sum(i -> norm(Ao_list[i] - A_list[i]), 1:2) < 1e-5
end

@testset "identify disc A Identical Noisy" begin
    prob = HSI.IdentifyProblemX(2, 1, 2, n_obs, u_seq, y1_seq, y2_seq)
    μA = zeros(SMatrix{2,2})
    δA = 10.0
    ϵ = noise*(1 + maximum(A -> opnorm(A, Inf), A_list)) + eps(10.0)
    qo_seq, Ao_list, io_coll, flag = HSI.identify_disc_model(
        prob, VLIN, Val(HSI.BKnown), VPId,
        μA, δA, B, ϵ, solver, Dbin=10)
    qo_seq = 3 .- qo_seq
    Ao_list[1], Ao_list[2] = Ao_list[2], Ao_list[1]
    @test flag
    @test isempty(io_coll)
    @test norm(qo_seq - q_seq) < 1e-14
    @test sum(i -> norm(Ao_list[i] - A_list[i]), 1:2) < 0.02
end

@testset "identify disc A Identical Noisy" begin
    x2_pert_seq = copy(x2_seq)
    x2_pert_seq[5] = 100*ones(SVector{2,Float64})
    prob = HSI.IdentifyProblemX(2, 1, 2, n_obs, u_seq, x1_seq, x2_pert_seq)
    μA = zeros(SMatrix{2,2})
    δA = 10.0
    ϵ = 1e-6
    qo_seq, Ao_list, io_coll, flag = HSI.identify_disc_model(
        prob, VLIN, Val(HSI.BKnown), VPId,
        μA, δA, B, ϵ, solver, Dbin=10)
    qo_seq = 3 .- qo_seq
    Ao_list[1], Ao_list[2] = Ao_list[2], Ao_list[1]
    @test flag
    @test io_coll == [5]
    @test qo_seq[5] == 3
    qo_seq[5] = q_seq[5]
    @test norm(qo_seq - q_seq) < 1e-14
    @test sum(i -> norm(Ao_list[i] - A_list[i]), 1:2) < 1e-5
end

sleep(0.1) # used for good printing
println("End test")

end  # module TestMain