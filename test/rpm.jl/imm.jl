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

sampler = SMC(1000)
mf = crpimm(data, rpm)

# Compute empirical posterior distribution over partitions
samples = sample(mf, sampler)

# Check that there is no NaN value associated
@test sum([sum(sample[:x][sample[:z]].== NaN)+sum(sample[:V][sample[:z]].== NaN) for sample in samples]) == 0


