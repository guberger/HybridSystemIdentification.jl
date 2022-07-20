module TestMain

using Test
using LinearAlgebra
using StaticArrays
using JuMP
using GLPK
# using Gurobi
@static if isdefined(Main, :TestLocal)
    include("../../src/HybridSystemIdentification.jl")
else
    using HybridSystemIdentification
end
HSI = HybridSystemIdentification

sleep(0.1) # used for good printing
println("Started test")

solver = optimizer_with_attributes(GLPK.Optimizer, "msg_lev"=>GLP_MSG_ON)
# solver = optimizer_with_attributes(Gurobi.Optimizer)

#===============================================================================
DATA
===============================================================================#
n_eq = 46
n_mode = 3
noise = 0.1

X1 = @SMatrix [1.0 2.0 3.0; 0.0 0.0 0.0]
X2 = @SMatrix [0.0 0.0 0.0; 1.0 2.0 3.0]
X3 = @SMatrix [3.0 2.0 1.0; 0.0 4.0 0.0]
X_list = (X1, X2, X3)

a_seq = Vector{SVector{3,Float64}}(undef, n_eq)
b_seq = Vector{SVector{2,Float64}}(undef, n_eq)

fnoise(q, i) = mod(sqrt(i*i + q), 1) > 0.5 ? noise : -noise

for i = 1:n_eq
    q = mod(i - 1, n_mode) + 1
    a_seq[i] = @SVector [-sqrt(i), cos(i/5), (i*i)^(1/3)]
    b_seq[i] = X_list[q]*a_seq[i] .+ fnoise(q, i)
end

Cbin = 100

#===============================================================================
TESTS
===============================================================================#
@testset "identify A feasible" begin
    Xlb = -10*ones(SMatrix{2,3})
    Xub = +10*ones(SMatrix{2,3})
    ϵ = noise
    Xo, io_coll, flag = HSI._max_FS(a_seq, b_seq, n_eq:-1:1,
        Xlb, Xub, ϵ, solver, Cbin)
    @test flag
    @test Set(io_coll) == Set(filter(i -> mod(i - 1, n_mode) == 0, 1:n_eq))
    @test norm(Xo - X_list[1], Inf) < 0.3
end

@testset "identify A infeasible" begin
    Xlb = 10*ones(SMatrix{2,3})
    Xub = 10*ones(SMatrix{2,3})
    ϵ = noise
    Xo, io_coll, flag = HSI._max_FS(a_seq, b_seq, n_eq:-1:1,
        Xlb, Xub, ϵ, solver, Cbin)
    @test flag
    @test isempty(io_coll)
    @test norm(Xo - 10*ones(SMatrix{2,3})) < eps(10.0)
end

sleep(0.1) # used for good printing
println("End test")

end  # module TestMain