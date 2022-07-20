module TestMain

using LinearAlgebra
using Random
using StaticArrays
using PyPlot
using JuMP
using Gurobi
include("../src/HybridSystemIdentification.jl")
HSI = HybridSystemIdentification

Random.seed!(0)

matplotlib.rc("legend", fontsize = 20)
matplotlib.rc("axes", labelsize = 20)
matplotlib.rc("xtick", labelsize = 15)
matplotlib.rc("ytick", labelsize = 15)
matplotlib.rc("text", usetex = true)

solver = optimizer_with_attributes(Gurobi.Optimizer)

sleep(0.1) # used for good printing
println("Started example simple")

#===============================================================================
DATA
===============================================================================#
D = 4
E = 1
n_mode = 2
α = (2π)*0.2
β = 0.1
A1 = @SMatrix [1.0 0.0   β 0.0;
               0.0 1.0 0.0   β;
               0.0 0.0 1.0 0.0;
               0.0 0.0 0.0 1.0]
A2 = @SMatrix [1.0 0.0    0.0     0.0;
               0.0 1.0    0.0     0.0;
               0.0 0.0 cos(α) -sin(α);
               0.0 0.0 sin(α)  cos(α)]
B = @SMatrix [0.0; 0.0; 0.0; 0.0]
A_list = (A1, A2)

n_obs = 100
noise = 0.01
p_turn = 0.2
max_rot = 4
pos0 = rand(2)
dir0 = normalize(rand(2))
x0 = SVector{4}(pos0..., dir0...)
q_seq = Vector{Int}(undef, n_obs)
u_seq = [SVector(1.0) for i = 1:n_obs]
x1_seq = Vector{SVector{4,Float64}}(undef, n_obs)
y1_seq = Vector{SVector{4,Float64}}(undef, n_obs)
x2_seq = Vector{SVector{4,Float64}}(undef, n_obs)
y2_seq = Vector{SVector{4,Float64}}(undef, n_obs)
y_seq = Vector{SVector{4,Float64}}(undef, n_obs + 1)
x = x0
q = rand(1:n_mode)

fnoise() = noise*SVector{4}(randn(4))

n_rot = rand(0:max_rot)

for i = 1:n_obs
    global n_rot
    if n_rot > 0
        q_seq[i] = 2
        n_rot -= 1
    elseif rand() < p_turn
        n_rot = rand(1:max_rot)
        q_seq[i] = 2
        n_rot -= 1
    else
        q_seq[i] = 1
    end
    A = A_list[q_seq[i]]
    x1_seq[i] = x
    y1_seq[i] = x1_seq[i] + fnoise()
    x2_seq[i] = A*x + B*u_seq[i] + fnoise()
    y2_seq[i] = x2_seq[i] + fnoise()
    global x = x2_seq[i]
    y_seq[i] = y1_seq[i]
end

y_seq[n_obs + 1] = y2_seq[end]

#===============================================================================
COMPUTE and PLOT
===============================================================================#
fig = figure(0, figsize=(10.5, 9.5))
gs = matplotlib.gridspec.GridSpec(2, 1, figure=fig,
    height_ratios=(3, 1), hspace=0.35)
ax = fig.add_subplot(get(gs, 0), aspect="equal")
ax_sw = fig.add_subplot(get(gs, 1))
ax.plot(y_seq[1][1:2]..., marker="o", ms=20, mec="gray", mew=3, mfc="none")
for i = 1:n_obs
    p1 = (y_seq[i][1], y_seq[i + 1][1])
    p2 = (y_seq[i][2], y_seq[i + 1][2])
    the_ls = q_seq[i] == 1 ? "solid" : "dashed"
    ax.plot(p1, p2, ls=the_ls, lw=1.5, c="black",
        marker=".", ms=10, mec="black", mfc="black")
    ax_sw.plot(i - 1, q_seq[i], ls="none", marker=".", ms=10, c="black")
end

prob = HSI.IdentifyProblemX(D, E, n_mode, n_obs, u_seq, x1_seq, x2_seq)
μA = zeros(SMatrix{4,4})

σA = 10.0
@time qo_seq, Ao_list = HSI.identify_disc_model(
    prob, Val(HSI.L2Norm), Val(HSI.BKnown), Val(HSI.PriorIdentical),
    μA, σA, B, nothing, print_period=1)
println(qo_seq - q_seq)
println(Ao_list)

# δA = 10.0
# ϵ = 0.01
# @time qo_seq, Ao_list = HSI.identify_disc_model(
#     prob, Val(HSI.L∞Norm), Val(HSI.BKnown), Val(HSI.PriorIdentical),
#     μA, δA, B, ϵ, solver, Dbin=10)
# println(qo_seq - q_seq)
# println(Ao_list)

prob = HSI.IdentifyProblemY(D, E, n_mode, n_obs + 1, u_seq, y_seq)
σd = noise
σn = noise
μA_list = [μA for q = 1:prob.n_mode]
σA_list = [σA for q = 1:prob.n_mode]
B_list = [B for q = 1:prob.n_mode]
@time Ao_list, xo_seq = HSI.identify_state_model(prob, Val(HSI.BKnown),
    q_seq, μA_list, σA_list, B_list, σd, σn,
    y_seq, Ao_list, print_period=1_000, tol_diff=1e-5)

for i = 1:n_obs
    p1 = (xo_seq[i][1], xo_seq[i + 1][1])
    p2 = (xo_seq[i][2], xo_seq[i + 1][2])
    the_ls = qo_seq[i] == 1 ? "solid" : "dashed"
    ax.plot(p1, p2, ls=the_ls, lw=1.5, c="red",
            marker=".", ms=10, mec="red", mfc="red")
end

ax.set_xlabel("position horizontal")
ax.set_ylabel("position vertical")
ax_sw.set_xlabel("time")
ax_sw.set_ylabel("mode")

LH = (
    matplotlib.lines.Line2D([0], [0], c="k", ls="-.", marker=".", ms=15,
        label="observation"),
    matplotlib.lines.Line2D([0], [0], c="red", ls="-.", marker=".", ms=15,
        label="estimation"),
    matplotlib.lines.Line2D([0], [0], ls="none", marker="o", ms=15, mec="gray",
        mew=3, mfc="none", label=L"\bar{x}(0)"),
    )
ax.legend(handles=LH)

println(Ao_list)
println(A_list)

fig.savefig(string("./figures/fig_exa_wandering_2modes.png"), dpi=200,
    transparent=false, bbox_inches="tight")

sleep(0.1) # used for good printing
println("End example simple")

end  # module TestMain