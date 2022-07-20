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
VL2N = Val(HSI.L2Norm)
VPId = Val(HSI.PriorIdentical)
VPDi = Val(HSI.PriorDifferent)

sleep(0.1) # used for good printing
println("Started test")

#===============================================================================
DATA
===============================================================================#
α = 0.1*(2*π)
β = 0.5
γ = 0.8
A1 = @SMatrix [β*cos(α) -β*sin(α); β*sin(α) β*cos(α)]
A2 = @SMatrix [γ 1.0; 0.0 γ]
B1 = @SMatrix [0.0; 1.0]
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
    q = mod(i - 1, 4) == 0 ? 1 : 2
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

fcurr_mode(q1, q2) = q1 == q2 ? -noise*log(0.5) : -noise*log(0.5)
fnext_mode(dist) = -noise*dist*max(log(0.5), log(0.5))
f_mode(dist, q1, q2) = (fcurr_mode(q1, q2), fnext_mode(dist))

#===============================================================================
TESTS
===============================================================================#
@testset "identify disc A Identical" begin
    prob = HSI.IdentifyProblemX(2, 1, 2, n_obs, u_seq, x1_seq, x2_seq)
    μA = zeros(SMatrix{2,2})
    σA = 10.0
    qo_seq, Ao_list = HSI.identify_disc_model(
        prob, VL2N, Val(HSI.BKnown), VPId,
        μA, σA, B_list[1], nothing, print_period=50)
    @test norm(qo_seq - q_seq) < 1e-14
    @test sum(i -> norm(Ao_list[i] - A_list[i]), 1:2) < 1e-5
    prob = HSI.IdentifyProblemX(2, 1, 2, n_obs, u_seq, y1_seq, y2_seq)
    qo_seq, Ao_list = HSI.identify_disc_model(
        prob, VL2N, Val(HSI.BKnown), VPId,
        μA, σA, B_list[1], nothing, print_period=50)
    @test norm(qo_seq - q_seq) < 1e-14
    @test sum(i -> norm(Ao_list[i] - A_list[i]), 1:2) < 0.004
end

@testset "identify disc A Identical f_mode" begin
    prob = HSI.IdentifyProblemX(2, 1, 2, n_obs, u_seq, x1_seq, x2_seq)
    μA = zeros(SMatrix{2,2})
    σA = 10.0
    qo_seq, Ao_list = HSI.identify_disc_model(
        prob, VL2N, Val(HSI.BKnown), VPId,
        μA, σA, B_list[1], f_mode, print_period=50)
    @test norm(qo_seq - q_seq) < 1e-14
    @test sum(i -> norm(Ao_list[i] - A_list[i]), 1:2) < 1e-5
    prob = HSI.IdentifyProblemX(2, 1, 2, n_obs, u_seq, y1_seq, y2_seq)
    qo_seq, Ao_list = HSI.identify_disc_model(
        prob, VL2N, Val(HSI.BKnown), VPId,
        μA, σA, B_list[1], f_mode, print_period=50)
    @test norm(qo_seq - q_seq) < 1e-14
    @test sum(i -> norm(Ao_list[i] - A_list[i]), 1:2) < 0.004
end

@testset "identify disc A Different 1" begin
    prob = HSI.IdentifyProblemX(2, 1, 2, n_obs, u_seq, x1_seq, x2_seq)
    μA_list = [zeros(SMatrix{2,2}) for q = 1:prob.n_mode]
    σA_list = [10.0 for q = 1:prob.n_mode]
    qo_seq, Ao_list = HSI.identify_disc_model(
        prob, VL2N, Val(HSI.BKnown), VPDi,
        μA_list, σA_list, B_list, nothing, print_period=50)
    if qo_seq[1] != q_seq[1]
        qo_seq = 3 .- qo_seq
        Ao_list[1], Ao_list[2] = Ao_list[2], Ao_list[1]
    end
    @test norm(qo_seq - q_seq) < 1e-14
    @test sum(i -> norm(Ao_list[i] - A_list[i]), 1:2) < 1e-5
    prob = HSI.IdentifyProblemX(2, 1, 2, n_obs, u_seq, y1_seq, y2_seq)
    qo_seq, Ao_list = HSI.identify_disc_model(
        prob, VL2N, Val(HSI.BKnown), VPDi,
        μA_list, σA_list, B_list, nothing, print_period=50)
    if qo_seq[1] != q_seq[1]
        qo_seq = 3 .- qo_seq
        Ao_list[1], Ao_list[2] = Ao_list[2], Ao_list[1]
    end
    @test norm(qo_seq - q_seq) < 1e-14
    @test sum(i -> norm(Ao_list[i] - A_list[i]), 1:2) < 0.004
end

@testset "identify disc A Different 1 f_mode" begin
    prob = HSI.IdentifyProblemX(2, 1, 2, n_obs, u_seq, x1_seq, x2_seq)
    μA_list = [zeros(SMatrix{2,2}) for q = 1:prob.n_mode]
    σA_list = [10.0 for q = 1:prob.n_mode]
    qo_seq, Ao_list = HSI.identify_disc_model(
        prob, VL2N, Val(HSI.BKnown), VPDi,
        μA_list, σA_list, B_list, f_mode, print_period=50)
    if qo_seq[1] != q_seq[1]
        qo_seq = 3 .- qo_seq
        Ao_list[1], Ao_list[2] = Ao_list[2], Ao_list[1]
    end
    @test norm(qo_seq - q_seq) < 1e-14
    @test sum(i -> norm(Ao_list[i] - A_list[i]), 1:2) < 1e-5
    prob = HSI.IdentifyProblemX(2, 1, 2, n_obs, u_seq, y1_seq, y2_seq)
    qo_seq, Ao_list = HSI.identify_disc_model(
        prob, VL2N, Val(HSI.BKnown), VPDi,
        μA_list, σA_list, B_list, f_mode, print_period=50)
    if qo_seq[1] != q_seq[1]
        qo_seq = 3 .- qo_seq
        Ao_list[1], Ao_list[2] = Ao_list[2], Ao_list[1]
    end
    @test norm(qo_seq - q_seq) < 1e-14
    @test sum(i -> norm(Ao_list[i] - A_list[i]), 1:2) < 0.004
end

@testset "identify disc A Different 2" begin
    prob = HSI.IdentifyProblemX(2, 1, 2, n_obs, u_seq, x1_seq, x2_seq)
    μA_list = [-ones(SMatrix{2,2}), ones(SMatrix{2,2})]
    σA_list = [10.0 for q = 1:prob.n_mode]
    qo_seq, Ao_list = HSI.identify_disc_model(
        prob, VL2N, Val(HSI.BKnown), VPDi,
        μA_list, σA_list, B_list, nothing, print_period=50)
    if qo_seq[1] != q_seq[1]
        qo_seq = 3 .- qo_seq
        Ao_list[1], Ao_list[2] = Ao_list[2], Ao_list[1]
    end
    @test norm(qo_seq - q_seq) < 1e-14
    @test sum(i -> norm(Ao_list[i] - A_list[i]), 1:2) < 1e-3
    prob = HSI.IdentifyProblemX(2, 1, 2, n_obs, u_seq, y1_seq, y2_seq)
    qo_seq, Ao_list = HSI.identify_disc_model(
        prob, VL2N, Val(HSI.BKnown), VPDi,
        μA_list, σA_list, B_list, nothing, print_period=50)
    if qo_seq[1] != q_seq[1]
        qo_seq = 3 .- qo_seq
        Ao_list[1], Ao_list[2] = Ao_list[2], Ao_list[1]
    end
    @test norm(qo_seq - q_seq) < 1e-14
    @test sum(i -> norm(Ao_list[i] - A_list[i]), 1:2) < 0.004
end

@testset "identify disc A Different 2 f_mode" begin
    prob = HSI.IdentifyProblemX(2, 1, 2, n_obs, u_seq, x1_seq, x2_seq)
    μA_list = [-ones(SMatrix{2,2}), ones(SMatrix{2,2})]
    σA_list = [10.0 for q = 1:prob.n_mode]
    qo_seq, Ao_list = HSI.identify_disc_model(
        prob, VL2N, Val(HSI.BKnown), VPDi,
        μA_list, σA_list, B_list, f_mode, print_period=50)
    if qo_seq[1] != q_seq[1]
        qo_seq = 3 .- qo_seq
        Ao_list[1], Ao_list[2] = Ao_list[2], Ao_list[1]
    end
    @test norm(qo_seq - q_seq) < 1e-14
    @test sum(i -> norm(Ao_list[i] - A_list[i]), 1:2) < 1e-3
    prob = HSI.IdentifyProblemX(2, 1, 2, n_obs, u_seq, y1_seq, y2_seq)
    qo_seq, Ao_list = HSI.identify_disc_model(
        prob, VL2N, Val(HSI.BKnown), VPDi,
        μA_list, σA_list, B_list, f_mode, print_period=50)
    if qo_seq[1] != q_seq[1]
        qo_seq = 3 .- qo_seq
        Ao_list[1], Ao_list[2] = Ao_list[2], Ao_list[1]
    end
    @test norm(qo_seq - q_seq) < 1e-14
    @test sum(i -> norm(Ao_list[i] - A_list[i]), 1:2) < 0.004
end

#===============================================================================
DATA
===============================================================================#
α = 0.1*(2*π)
β = 0.5
γ = 0.8
A1 = A2 = @SMatrix [β*cos(α) -β*sin(α); β*sin(α) β*cos(α)]
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
    q = mod(i - 1, 4) == 0 ? 1 : 2
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
@testset "identify disc B Identical" begin
    prob = HSI.IdentifyProblemX(2, 1, 2, n_obs, u_seq, x1_seq, x2_seq)
    μB = zeros(SMatrix{2,1})
    σB = 10.0
    qo_seq, Bo_list = HSI.identify_disc_model(
        prob, VL2N, Val(HSI.AKnown), VPId,
        A_list[1], μB, σB, nothing, print_period=50)
    @test norm(qo_seq - q_seq) < 1e-14
    @test sum(i -> norm(Bo_list[i] - B_list[i]), 1:2) < 1e-5
    prob = HSI.IdentifyProblemX(2, 1, 2, n_obs, u_seq, y1_seq, y2_seq)
    qo_seq, Bo_list = HSI.identify_disc_model(
        prob, VL2N, Val(HSI.AKnown), VPId,
        A_list[1], μB, σB, nothing, print_period=50)
    @test norm(qo_seq - q_seq) < 1e-14
    @test sum(i -> norm(Bo_list[i] - B_list[i]), 1:2) < 0.009
end

f_mode(q1, q2) = q1 == q2 ? 0.0 : 0.001

@testset "identify disc B Identical f_mode" begin
    prob = HSI.IdentifyProblemX(2, 1, 2, n_obs, u_seq, x1_seq, x2_seq)
    μB = zeros(SMatrix{2,1})
    σB = 10.0
    qo_seq, Bo_list = HSI.identify_disc_model(
        prob, VL2N, Val(HSI.AKnown), VPId,
        A_list[1], μB, σB, f_mode, print_period=50)
    @test norm(qo_seq - q_seq) < 1e-14
    @test sum(i -> norm(Bo_list[i] - B_list[i]), 1:2) < 1e-5
    prob = HSI.IdentifyProblemX(2, 1, 2, n_obs, u_seq, y1_seq, y2_seq)
    qo_seq, Bo_list = HSI.identify_disc_model(
        prob, VL2N, Val(HSI.AKnown), VPId,
        A_list[1], μB, σB, f_mode, print_period=50)
    @test norm(qo_seq - q_seq) < 1e-14
    @test sum(i -> norm(Bo_list[i] - B_list[i]), 1:2) < 0.009
end

@testset "identify disc B Different 1" begin
    prob = HSI.IdentifyProblemX(2, 1, 2, n_obs, u_seq, x1_seq, x2_seq)
    μB_list = [zeros(SMatrix{2,1}) for q = 1:prob.n_mode]
    σB_list = [10.0 for q = 1:prob.n_mode]
    qo_seq, Bo_list = HSI.identify_disc_model(
        prob, VL2N, Val(HSI.AKnown), VPDi,
        A_list, μB_list, σB_list, nothing, print_period=50)
    @test norm(qo_seq - q_seq) < 1e-14
    @test sum(i -> norm(Bo_list[i] - B_list[i]), 1:2) < 1e-5
    prob = HSI.IdentifyProblemX(2, 1, 2, n_obs, u_seq, y1_seq, y2_seq)
    qo_seq, Bo_list = HSI.identify_disc_model(
        prob, VL2N, Val(HSI.AKnown), VPDi,
        A_list, μB_list, σB_list, nothing, print_period=50)
    @test norm(qo_seq - q_seq) < 1e-14
    @test sum(i -> norm(Bo_list[i] - B_list[i]), 1:2) < 0.009
end

f_mode(q1, q2) = q1 == q2 ? 0.0 : 0.00001

@testset "identify disc B Different 1 f_mode" begin
    prob = HSI.IdentifyProblemX(2, 1, 2, n_obs, u_seq, x1_seq, x2_seq)
    μB_list = [zeros(SMatrix{2,1}) for q = 1:prob.n_mode]
    σB_list = [10.0 for q = 1:prob.n_mode]
    qo_seq, Bo_list = HSI.identify_disc_model(
        prob, VL2N, Val(HSI.AKnown), VPDi,
        A_list, μB_list, σB_list, f_mode, print_period=50)
    @test norm(qo_seq - q_seq) < 1e-14
    @test sum(i -> norm(Bo_list[i] - B_list[i]), 1:2) < 1e-5
    prob = HSI.IdentifyProblemX(2, 1, 2, n_obs, u_seq, y1_seq, y2_seq)
    qo_seq, Bo_list = HSI.identify_disc_model(
        prob, VL2N, Val(HSI.AKnown), VPDi,
        A_list, μB_list, σB_list, f_mode, print_period=50)
    @test norm(qo_seq - q_seq) < 1e-14
    @test sum(i -> norm(Bo_list[i] - B_list[i]), 1:2) < 0.009
end

@testset "identify disc B Different 2" begin
    prob = HSI.IdentifyProblemX(2, 1, 2, n_obs, u_seq, x1_seq, x2_seq)
    μB_list = [SMatrix{2,1}(0.5, 0.0), SMatrix{2,1}(0.0, 0.5)]
    σB_list = [10.0 for q = 1:prob.n_mode]
    qo_seq, Bo_list = HSI.identify_disc_model(
        prob, VL2N, Val(HSI.AKnown), VPDi,
        A_list, μB_list, σB_list, nothing, print_period=50)
    @test norm(qo_seq - q_seq) < 1e-14
    @test sum(i -> norm(Bo_list[i] - B_list[i]), 1:2) < 1e-3
    prob = HSI.IdentifyProblemX(2, 1, 2, n_obs, u_seq, y1_seq, y2_seq)
    qo_seq, Bo_list = HSI.identify_disc_model(
        prob, VL2N, Val(HSI.AKnown), VPDi,
        A_list, μB_list, σB_list, nothing, print_period=50)
    @test norm(qo_seq - q_seq) < 1e-14
    @test sum(i -> norm(Bo_list[i] - B_list[i]), 1:2) < 0.009
end

f_mode(q1, q2) = q1 == q2 ? 0.0 : 0.0001

@testset "identify disc B Different 2 f_mode" begin
    prob = HSI.IdentifyProblemX(2, 1, 2, n_obs, u_seq, x1_seq, x2_seq)
    μB_list = [SMatrix{2,1}(0.5, 0.0), SMatrix{2,1}(0.0, 0.5)]
    σB_list = [10.0 for q = 1:prob.n_mode]
    qo_seq, Bo_list = HSI.identify_disc_model(
        prob, VL2N, Val(HSI.AKnown), VPDi,
        A_list, μB_list, σB_list, f_mode, print_period=50)
    @test norm(qo_seq - q_seq) < 1e-14
    @test sum(i -> norm(Bo_list[i] - B_list[i]), 1:2) < 1e-3
    prob = HSI.IdentifyProblemX(2, 1, 2, n_obs, u_seq, y1_seq, y2_seq)
    qo_seq, Bo_list = HSI.identify_disc_model(
        prob, VL2N, Val(HSI.AKnown), VPDi,
        A_list, μB_list, σB_list, f_mode, print_period=50)
    @test norm(qo_seq - q_seq) < 1e-14
    @test sum(i -> norm(Bo_list[i] - B_list[i]), 1:2) < 0.009
end

sleep(0.1) # used for good printing
println("End test")

end  # module TestMain