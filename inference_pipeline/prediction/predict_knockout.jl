using DifferentialEquations
using Plots
using Statistics
using LinearAlgebra

# Compute N dynamically based on true network binary size
true_file_size = filesize("results/W_true.bin")
N = Int(sqrt(true_file_size / sizeof(Float64)))
println("Detected N = ", N, " from W_true.bin.")

# Compute N_samples dynamically based on binary file size
file_size = filesize("results/W_ensemble.bin")
N_samples = div(file_size, N * N * sizeof(Float64))
println("Detected N_samples = ", N_samples, " from W_ensemble.bin.")

# Load the ensemble results from raw binary (C-order)
println("Loading ensemble results from binary...")

# Python saved as (N_samples, N, N). 
# Julia reads in Fortran order. First dimension in Julia varies fastest.
# Python's fastest varying dimension is the last one (size N).
raw_W = read!("results/W_ensemble.bin", Array{Float64}(undef, N, N, N_samples))

# Python saved as (N_samples, 3).
raw_theta = read!("results/theta_ensemble.bin", Array{Float64}(undef, 3, N_samples))

println("Loading TRUE network...")
W_true = permutedims(read!("results/W_true.bin", Array{Float64}(undef, N, N)), (2, 1))
theta_true = read!("results/theta_true.bin", Array{Float64}(undef, 3))

println("W_ensemble binary read complete.")

# Perturbation target: Knockout Master Regulator
# Read from command line arguments if provided, else default to 1 (Node 0 in Python)
ko_idx = length(ARGS) > 0 ? parse(Int, ARGS[1]) : 1

# Base setup for stationary state
tspan = (0.0, 10.0) # Integrate longer to ensure steady state
u0_base = rand(N) .* 0.9 .+ 0.1 # Random initial state between 0.1 and 1.0
u0_base[ko_idx] = 0.0           # Knockout initial state

# Preallocate for performance
function f!(du, u, p, t)
    W, d, V, D_base, ko = p
    
    # W * u
    mul!(du, W, u)
    
    # -d*u + V*sigmoid(W*u)
    @inbounds for i in 1:length(u)
        du[i] = -d * u[i] + V * (1.0 / (1.0 + exp(-du[i])))
    end
    
    # Enforce Knockout
    if ko > 0
        du[ko] = 0.0
    end
end

function g!(du, u, p, t)
    W, d, V, D_base, ko = p
    @inbounds for i in 1:length(u)
        du[i] = sqrt(D_base * max(u[i], 0.01))
    end
    
    if ko > 0
        du[ko] = 0.0
    end
end

# Define the Ensemble Problem
function prob_func(prob, i, repeat)
    # Extract the i-th network and its parameters
    # raw_W[j, k, i] corresponds to Python's W[i, k, j]. 
    # We want W_julia[k, j] = W_python[i, k, j].
    # So we transpose the slice!
    W = permutedims(raw_W[:, :, i], (2, 1))
    
    # Parameters
    # raw_theta[p, i] corresponds to Python's theta[i, p].
    d = exp(raw_theta[1, i])
    V = exp(raw_theta[2, i])
    D_base = exp(raw_theta[3, i])
    
    # Perturb the network (remove incoming/outgoing edges)
    W_pert = copy(W)
    W_pert[ko_idx, :] .= 0.0
    W_pert[:, ko_idx] .= 0.0
    
    p_new = (W_pert, d, V, D_base, ko_idx)
    
    # We use the SAME initial condition for all networks to isolate the network's effect
    remake(prob, p=p_new, u0=copy(u0_base))
end

# Define the unperturbed Ensemble Problem
function prob_func_unpert(prob, i, repeat)
    W = permutedims(raw_W[:, :, i], (2, 1))
    d = exp(raw_theta[1, i])
    V = exp(raw_theta[2, i])
    D_base = exp(raw_theta[3, i])
    
    p_new = (copy(W), d, V, D_base, 0) # ko=0 means no knockout
    remake(prob, p=p_new, u0=copy(u0_base))
end

# Dummy base problem
p_dummy = (zeros(N, N), 1.0, 1.0, 1.0, 0)
prob = SDEProblem(f!, g!, u0_base, tspan, p_dummy)
ensemble_prob = EnsembleProblem(prob, prob_func=prob_func)
ensemble_prob_unpert = EnsembleProblem(prob, prob_func=prob_func_unpert)

println("Simulating ensemble of ", N_samples, " perturbed networks...")
sim_pert = solve(ensemble_prob, SOSRI(), EnsembleThreads(), trajectories=N_samples, saveat=0.5)

println("Simulating ensemble of ", N_samples, " unperturbed networks...")
sim_unpert = solve(ensemble_prob_unpert, SOSRI(), EnsembleThreads(), trajectories=N_samples, saveat=0.5)

# Simulate TRUE network
# Simulate TRUE network (Perturbed and Unperturbed)
println("Simulating TRUE networks (Perturbed and Unperturbed)...")
W_true_pert = copy(W_true)
W_true_pert[ko_idx, :] .= 0.0
W_true_pert[:, ko_idx] .= 0.0

p_true_pert = (W_true_pert, exp(theta_true[1]), exp(theta_true[2]), exp(theta_true[3]), ko_idx)
p_true_unpert = (copy(W_true), exp(theta_true[1]), exp(theta_true[2]), exp(theta_true[3]), 0)

prob_true_pert = SDEProblem(f!, g!, u0_base, tspan, p_true_pert)
prob_true_unpert = SDEProblem(f!, g!, u0_base, tspan, p_true_unpert)

ensemble_prob_true_pert = EnsembleProblem(prob_true_pert)
ensemble_prob_true_unpert = EnsembleProblem(prob_true_unpert)

sim_true_pert = solve(ensemble_prob_true_pert, SOSRI(), EnsembleThreads(), trajectories=100, saveat=0.5)
sim_true_unpert = solve(ensemble_prob_true_unpert, SOSRI(), EnsembleThreads(), trajectories=100, saveat=0.5)

println("Extracting stationary states...")
stat_inferred_pert = hcat([sol.u[end] for sol in sim_pert]...)
stat_inferred_unpert = hcat([sol.u[end] for sol in sim_unpert]...)
stat_true_pert = hcat([sol.u[end] for sol in sim_true_pert]...)
stat_true_unpert = hcat([sol.u[end] for sol in sim_true_unpert]...)

# Compute Mean Delta and Std Delta
mean_inf_delta = mean(stat_inferred_pert, dims=2)[:, 1] .- mean(stat_inferred_unpert, dims=2)[:, 1]
std_inf_delta = sqrt.(var(stat_inferred_pert, dims=2)[:, 1] .+ var(stat_inferred_unpert, dims=2)[:, 1])

mean_true_delta = mean(stat_true_pert, dims=2)[:, 1] .- mean(stat_true_unpert, dims=2)[:, 1]
std_true_delta = sqrt.(var(stat_true_pert, dims=2)[:, 1] .+ var(stat_true_unpert, dims=2)[:, 1])


println("Plotting Delta Stationary State distributions...")
p = scatter(mean_true_delta, mean_inf_delta, 
            xerror=std_true_delta, yerror=std_inf_delta,
            xlabel="TRUE Δ Stationary State", 
            ylabel="INFERRED Ensemble Δ Stationary State",
            title="Shift in Steady State after Gene $ko_idx Knockout",
            label="Genes", color=:blue, alpha=0.6,
            guidefontsize=10, titlefontsize=12)

# Add diagonal reference line
min_val = min(minimum(mean_true_delta), minimum(mean_inf_delta)) - 0.1
max_val = max(maximum(mean_true_delta), maximum(mean_inf_delta)) + 0.1
plot!(p, [min_val, max_val], [min_val, max_val], color=:red, linestyle=:dash, label="Perfect Match", lw=2)

# Add crosshairs for 0,0
plot!(p, [min_val, max_val], [0, 0], color=:gray, linestyle=:dot, label="")
plot!(p, [0, 0], [min_val, max_val], color=:gray, linestyle=:dot, label="")

savefig("images/delta_stationary_states_plot.png")
println("Saved images/delta_stationary_states_plot.png successfully!")
