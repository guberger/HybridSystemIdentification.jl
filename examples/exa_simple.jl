module TestMain

using LinearAlgebra
using Random
using StaticArrays
using PyPlot
include("../src/HybridSystemIdentification.jl")
HSI = HybridSystemIdentification

matplotlib.rc("legend", fontsize = 20)
matplotlib.rc("axes", labelsize = 20)
matplotlib.rc("xtick", labelsize = 15)
matplotlib.rc("ytick", labelsize = 15)
matplotlib.rc("text", usetex = true)

sleep(0.1) # used for good printing
println("Started example simple")

#===============================================================================
DATA
===============================================================================#
D = 2
E = 1
n_mode = 2
α = 0.01*(2*π)
β = 0.9
A1 = @SMatrix [cos(α) -β*sin(α); 1/β*sin(α) cos(α)]
A2 = @SMatrix [cos(α) 1/β*sin(α); -β*sin(α) cos(α)]
B1 = @SMatrix [0.0; 0.0]
B2 = @SMatrix [0.0; 0.0]
A_list = (A1, A2)
B_list = (B1, B2)

n_obs = 100
noise = 0.01
x0 = @SVector [1.0, -1.0]
q_seq = Vector{Int}(undef, n_obs)
u_seq = [SVector(0.0) for i = 1:n_obs]
x1_seq = Vector{SVector{2,Float64}}(undef, n_obs)
y1_seq = Vector{SVector{2,Float64}}(undef, n_obs)
x2_seq = Vector{SVector{2,Float64}}(undef, n_obs)
y2_seq = Vector{SVector{2,Float64}}(undef, n_obs)
y_seq = Vector{SVector{2,Float64}}(undef, n_obs + 1)
x = x0
Random.seed!(0)

fnoise() = noise*SVector(randn(), randn())

for i = 1:n_obs
    q = mod((i - 1)/10, 2) < 1 ? 1 : 2
    q_seq[i] = q
    A = A_list[q_seq[i]]; B = B_list[q_seq[i]]
    x1_seq[i] = x
    y1_seq[i] = x1_seq[i] + fnoise()
    x2_seq[i] = A*x + B*u_seq[i] + fnoise()
    y2_seq[i] = x2_seq[i] + fnoise()
    global x = x2_seq[i]
    y_seq[i] = y1_seq[i]
end

y_seq[n_obs + 1] = y2_seq[end]

p_sw = 0.25
fcurr_mode(q1, q2) = q1 == q2 ? -noise*log(1 - p_sw) : -noise*log(p_sw)
fnext_mode(dist) = -noise*dist*max(log(1 - p_sw), log(p_sw))
f_mode(dist, q1, q2) = (fcurr_mode(q1, q2), fnext_mode(dist))

#===============================================================================
COMPUTE and PLOT
===============================================================================#
fig = figure(0, figsize=(10.5, 9.5))
gs = matplotlib.gridspec.GridSpec(2, 1, figure=fig,
    height_ratios=(3, 1), hspace=0.2)
ax = fig.add_subplot(get(gs, 0))
ax_sw = fig.add_subplot(get(gs, 1))
ax.plot(y_seq[1]..., marker="o", ms=20, mec="gray", mew=3, mfc="none")
for i = 1:n_obs
    p1 = (y_seq[i][1], y_seq[i + 1][1])
    p2 = (y_seq[i][2], y_seq[i + 1][2])
    the_ls = q_seq[i] == 1 ? "solid" : "dashed"
    ax.plot(p1, p2, ls=the_ls, lw=1.5, c="black",
        marker=".", ms=10, mec="black", mfc="black")
    ax_sw.plot(i - 1, q_seq[i], ls="none", marker=".", ms=10, c="black")
end

prob = HSI.IdentifyProblemX(2, 1, 2, n_obs, u_seq, x1_seq, x2_seq)
μA = zeros(SMatrix{2,2})
σA = 100.0
@time qo_seq, Ao_list = HSI.identify_disc_model(
    prob, Val(HSI.L2Norm), Val(HSI.BKnown), Val(HSI.PriorIdentical),
    μA, σA, B_list[1], f_mode, print_period=Inf)
println(qo_seq - q_seq)
println(Ao_list)

μA_list = [μA for q = 1:prob.n_mode]
σA_list = [σA for q = 1:prob.n_mode]
@time qo_seq, Ao_list = HSI.identify_disc_model(
    prob, Val(HSI.L2Norm), Val(HSI.BKnown), Val(HSI.PriorDifferent),
    μA_list, σA_list, B_list, f_mode, print_period=Inf)

println(qo_seq - q_seq)
println(Ao_list)

prob = HSI.IdentifyProblemY(2, 1, 2, n_obs + 1, u_seq, y_seq)
σd = noise
σn = noise
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

fig.savefig(string("./figures/fig_exa_simple.png"), dpi=200,
    transparent=false, bbox_inches="tight")

sleep(0.1) # used for good printing
println("End example simple")

end  # module TestMain