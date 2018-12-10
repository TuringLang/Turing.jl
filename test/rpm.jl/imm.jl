using Turing
using Test
using Random

Random.seed!(12)

data = vcat(rand(Normal(0, 0.5), 10), rand(Normal(8, 0.5), 10))

@model crpimm(y;
              α = 0.1,
              H = Normal(mean(y), std(y) * 2),
              rpm = DirichletProcess(α)
             ) = begin
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
            x[z[i]] ~ H
        end
        cluster_counts[z[i]] += 1

        # Draw observation.
        y[i] ~ Normal(x[z[i]], 0.5)
    end
    return z, x
end

data = shuffle(data)
mf = crpimm(data)
chn = sample(mf, PG(50, 500))

k = map(length, map(unique, chn[:z]))
cs = map(i -> sum(k .== i), 1:maximum(k))

@test argmax(cs) == 2
