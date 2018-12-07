using Turing
using Test
using Random

Random.seed!(12)

data = vcat(randn(10), randn(10) .+ 10, randn(10) .- 10)

@model infiniteMixture(y) = begin

    # Number of observations.
    N = length(y)
    H = Normal(mean(y), 1.0)

    x = tzeros(Float64, N);
    z = tzeros(Int, N)

    K = 0

    α = 1.0
    rpm = DirichletProcess(α)
    cluster_counts = tzeros(Int, 0)
    for i in 1:N

        z[i] ~ ChineseRestaurantProcess(rpm, cluster_counts)
        if z[i] > K
            K = K + 1
            x[K] ~ H
            push!(cluster_counts, 1)
        else
            cluster_counts[z[i]] += 1
        end
        y[i] ~ Normal(x[z[i]], 1.0)
    end

    return z
end

mf = infiniteMixture(data)
chn = sample(mf, PG(50, 100))

@test_broken length(unique(last(chn[:z]))) == 3
