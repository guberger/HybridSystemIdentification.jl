module TestMain

using Test
using LinearAlgebra
using StaticArrays
@static if isdefined(Main, :TestLocal)
    include("../../src/HybridSystemIdentification.jl")
else
    using HybridSystemIdentification
end
HSI = HybridSystemIdentification

sleep(0.1) # used for good printing
println("Started test")

# Solution given by A1 = A2 = 0, x1 = 1, x2 = x3 = 0
@testset "Identify State A Simple 0" begin
    A0_list = ([1][:, :], [1][:, :])
    B_list = ([0], [0])
    x0_seq = [[-1], [-1], [-1]]
    u_seq = [[0], [0], [0]]
    y_seq = [[1], [0], [0]]
    q_seq = [1, 2]
    n_obs = 3
    prob = HSI.IdentifyProblemY(1, 1, 2, n_obs, u_seq, y_seq)
    μA_list = [zeros(SMatrix{1,1}) for q = 1:prob.n_mode]
    σA_list = [1.0 for q = 1:prob.n_mode]
    σd = 1.0
    σn = 0.5
    Ao_list, xo_seq = HSI.identify_state_model(prob, Val(HSI.BKnown),
        q_seq, μA_list, σA_list, B_list, σd, σn,
        x0_seq, A0_list, print_period=1_000, tol_diff=1e-8)
    @test sum(i -> norm(Ao_list[i]), 1:2) < 1e-6
    @test sum(i -> norm(xo_seq[i] - y_seq[i]), 1:n_obs) < 1e-6
end

# minimize (x-1)^2 + (y-1)^2 + (z*x-y)^2 + z^2
@testset "Identify State A Simple Wolfram 0" begin
    A0_list = ([1][:, :],)
    B_list = ([0],)
    x0_seq = [[-1], [-1]]
    u_seq = [[0], [0]]
    y_seq = [[1], [1]]
    q_seq = [1, 1]
    n_obs = 2
    prob = HSI.IdentifyProblemY(1, 1, 1, n_obs, u_seq, y_seq)
    μA_list = [zeros(SMatrix{1,1}) for q = 1:prob.n_mode]
    σA_list = [1.0 for q = 1:prob.n_mode]
    σd = 1.0
    σn = 1.0
    Ao_list, xo_seq = HSI.identify_state_model(prob, Val(HSI.BKnown),
        q_seq, μA_list, σA_list, B_list, σd, σn,
        x0_seq, A0_list, print_period=1_000, tol_diff=1e-8)
    @test norm(Ao_list[1] - [0.343166]) < 1e-5
    @test norm(xo_seq[1] - [1.10643]) < 1e-5
    @test norm(xo_seq[2] - [0.689845]) < 1e-5
end

# minimize (x-1)^2 + (y-1)^2 + (z*x-y)^2 + (z + 1)^2
@testset "Identify State A Simple Wolfram 1" begin
    A0_list = ([1][:, :],)
    B_list = ([0],)
    x0_seq = [[-1], [-1]]
    u_seq = [[0], [0]]
    y_seq = [[1], [1]]
    q_seq = [1, 1]
    n_obs = 2
    prob = HSI.IdentifyProblemY(1, 1, 1, n_obs, u_seq, y_seq)
    μA_list = [-ones(SMatrix{1,1}) for q = 1:prob.n_mode]
    σA_list = [1.0 for q = 1:prob.n_mode]
    σd = 1.0
    σn = 1.0
    Ao_list, xo_seq = HSI.identify_state_model(prob, Val(HSI.BKnown),
        q_seq, μA_list, σA_list, B_list, σd, σn,
        x0_seq, A0_list, print_period=1_000, tol_diff=1e-8)
    @test norm(Ao_list[1] - [-0.596072]) < 1e-5
    @test norm(xo_seq[1] - [0.596072]) < 1e-5
    @test norm(xo_seq[2] - [0.322349]) < 1e-5
end

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
A_noisy_list = (A1 .+ 0.1, A2 .- 0.2)
x_noisy_seq = map(x -> x .+ 0.1, x1_seq)

#===============================================================================
TESTS
===============================================================================#
@testset "Identify State A Data" begin
    prob = HSI.IdentifyProblemY(2, 1, 2, n_obs, u_seq, x1_seq)
    μA_list = [zeros(SMatrix{2,2}) for q = 1:prob.n_mode]
    σA_list = [1e20 for q = 1:prob.n_mode]
    σd = 1.0
    σn = 0.05
    Ao_list, xo_seq = HSI.identify_state_model(prob, Val(HSI.BKnown),
        q_seq, μA_list, σA_list, B_list, σd, σn,
        x_noisy_seq, A_noisy_list, print_period=1_000, tol_diff=1e-8)
    @test sum(i -> norm(Ao_list[i] - A_list[i]), 1:2) < 1e-4
    @test sum(i -> norm(xo_seq[i] - x1_seq[i]), 1:n_obs) < 1e-6
end

sleep(0.1) # used for good printing
println("End test")

end  # module TestMain