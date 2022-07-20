module HybridSystemIdentification

using LinearAlgebra
using Printf
using DataStructures
using JuMP

abstract type IdentifyProblem end

struct IdentifyProblemX{UT,XT} <: IdentifyProblem
    D::Int
    E::Int
    n_mode::Int
    n_obs::Int
    u_seq::Vector{UT}
    x1_seq::Vector{XT}
    x2_seq::Vector{XT}
end

struct IdentifyProblemY{UT,YT} <: IdentifyProblem
    D::Int
    E::Int
    n_mode::Int
    n_obs::Int
    u_seq::Vector{UT}
    y_seq::Vector{YT}
end

@enum OptimRule L2Norm L∞Norm PartitionFS

@enum ABInfo AKnown BKnown

@enum ModePrior PriorIdentical PriorDifferent

input_dim(prob::IdentifyProblem) = prob.E
state_dim(prob::IdentifyProblem) = prob.D

include("Identify_L2_norm/identify_model.jl")
include("Identify_L2_norm/identify_disc_model.jl")
include("Identify_L2_norm/identify_state_model.jl")

include("Identify_LInf_norm/identify_model.jl")
include("Identify_LInf_norm/identify_disc_model.jl")

end # module
