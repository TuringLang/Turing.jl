using Turing
using Turing.RandomMeasures
using Test
using Random

# Data
data = [-2,2,-1.5,1.5]

# Base distribution
mu_0 = mean(data)
sigma_0 = 4
sigma_1 = 0.5
tau0 = 1/sigma_0^2
tau1 = 1/sigma_1^2

# PYP parameters
sigma = 0.25 # = alpha
theta = 0.1

# DP parameters
alpha = 0.25

@model crpimm(y, rpm) = begin

    # Base distribution.
    H = Normal(mu_0, sigma_0)
    # Latent assignments.
    N = length(y)
    z = tzeros(Int, N)

    # Cluster counts.
    cluster_counts = tzeros(Int, N)

    # Cluster locations.
    x = tzeros(Float64, N)

    for i in 1:N
        # Draw assignments using a CRP.
        z[i] ~ ChineseRestaurantProcess(rpm, cluster_counts)
        if cluster_counts[z[i]] == 0
            # Cluster is new, therefore, draw new location.
            l ~ H
            x[z[i]] ~ H
        end
        cluster_counts[z[i]] += 1
        # Draw observation.
        y[i] ~ Normal(x[z[i]], sigma_1)
    end
end

rpm = DirichletProcess(alpha)

sampler = SMC(5000)
mf = crpimm(data, rpm)

# Compute empirical posterior distribution over partitions
samples = sample(mf, sampler)

# Check that there is no NaN value associated
z_samples = Int.(samples[:z])
@test all(!isnan, samples[:x][z_samples])
@test all(!ismissing, samples[:x][z_samples])

include("utils.jl")

partitions = [[[1, 2, 3, 4]], [[1, 2, 3], [4]], [[1, 2, 4], [3]], [[1, 2], [3, 4]], [[1, 2], [3], [4]],
    [[1, 3, 4], [2]], [[1, 3], [2, 4]], [[1, 3], [2], [4]], [[1, 4], [2, 3]], [[1], [2, 3, 4]],
    [[1], [2, 3], [4]], [[1, 4], [2], [3]], [[1], [2, 4], [3]], [[1], [2], [3, 4]], [[1], [2], [3], [4]]]

empirical_probs = zeros(length(partitions))
w = map(x -> x.weight, samples.info[:samples])
sum_weights = sum(w)
z = samples[:z]

for i in 1:size(z,1)
    partition = map(c -> findall(z[i,:,1] .== c), unique(z[i,:,1]))
    partition_idx = findfirst(p -> sort(p) == sort(partition), partitions)
    @test partition_idx != nothing
    empirical_probs[partition_idx] += sum_weights == 0 ? 1 : w[i]
end

if sum_weights == 0
    empirical_probs /= length(w)
end

l2, csqr = correct_posterior(empirical_probs, data, partitions, tau0, tau1)

@test l2 < 0.05
