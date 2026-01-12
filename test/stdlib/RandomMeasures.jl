module RandomMeasuresTests

using Distributions: Normal, sample
using Random: Random
using Test: @test, @testset
using Turing
using Turing.RandomMeasures: ChineseRestaurantProcess, DirichletProcess

@testset "RandomMeasures.jl" begin
    @testset "Infinite Mixture Model" begin
        @model function infiniteGMM(x)
            # Hyper-parameters, i.e. concentration parameter and parameters of H.
            α = 1.0
            μ0 = 0.0
            σ0 = 1.0

            # Define random measure, e.g. Dirichlet process.
            rpm = DirichletProcess(α)

            # Define the base distribution, i.e. expected value of the Dirichlet process.
            H = Normal(μ0, σ0)

            # Latent assignment.
            z = zeros(Int, length(x))

            # Locations of the infinitely many clusters.
            μ = zeros(Float64, 0)

            for i in 1:length(x)

                # Number of clusters.
                K = maximum(z)
                nk = Vector{Int}(map(k -> sum(z .== k), 1:K))

                # Draw the latent assignment.
                z[i] ~ ChineseRestaurantProcess(rpm, nk)

                # Create a new cluster?
                if z[i] > K
                    push!(μ, 0.0)

                    # Draw location of new cluster.
                    μ[z[i]] ~ H
                end

                # Draw observation.
                x[i] ~ Normal(μ[z[i]], 1.0)
            end
        end

        # Generate some test data.
        Random.seed!(1)
        data = vcat(randn(10), randn(10) .- 5, randn(10) .+ 10)
        data .-= mean(data)
        data /= std(data)

        # MCMC sampling
        Random.seed!(2)
        iterations = 500
        model_fun = infiniteGMM(data)
        chain = sample(model_fun, SMC(), iterations)

        @test chain isa VNChain
    end
    # partitions = [
    #     [[1, 2, 3, 4]],
    #     [[1, 2, 3], [4]],
    #     [[1, 2, 4], [3]],
    #     [[1, 2], [3, 4]],
    #     [[1, 2], [3], [4]],
    #     [[1, 3, 4], [2]],
    #     [[1, 3], [2, 4]],
    #     [[1, 3], [2], [4]],
    #     [[1, 4], [2, 3]],
    #     [[1], [2, 3, 4]],
    #     [[1], [2, 3], [4]],
    #     [[1, 4], [2], [3]],
    #     [[1], [2, 4], [3]],
    #     [[1], [2], [3, 4]],
    #     [[1], [2], [3], [4]]]

    # @testset "chinese restaurant processes" begin
    #     # Data
    #     data = [-2,2,-1.5,1.5]

    #     # Base distribution
    #     mu_0 = mean(data)
    #     sigma_0 = 4
    #     sigma_1 = 0.5
    #     tau0 = 1/sigma_0^2
    #     tau1 = 1/sigma_1^2

    #     # DP parameters
    #     alpha = 0.25

    #     @model function crpimm(y, rpm)
    #         # Base distribution.
    #         H = Normal(mu_0, sigma_0)
    #         # Latent assignments.
    #         N = length(y)
    #         z = tzeros(Int, N)

    #         # Cluster counts.
    #         cluster_counts = tzeros(Int, N)

    #         # Cluster locations.
    #         x = tzeros(Float64, N)
    #         for i in 1:N
    #             # Draw assignments using a CRP.
    #             z[i] ~ ChineseRestaurantProcess(rpm, cluster_counts)
    #             if cluster_counts[z[i]] == 0
    #                 # Cluster is new, therefore, draw new location.
    #                 x[z[i]] ~ H
    #             end
    #             cluster_counts[z[i]] += 1
    #             # Draw observation.
    #             y[i] ~ Normal(x[z[i]], sigma_1)
    #         end
    #     end

    #     rpm = DirichletProcess(alpha)

    #     sampler = SMC()
    #     mf = crpimm(data, rpm)

    #     # Compute empirical posterior distribution over partitions
    #     samples = sample(mf, sampler, 1000)

    #     # Check that there is no NaN value associated
    #     z_samples = Int.(samples[:z])
    #     @test all(!isnan, samples[:x][z_samples])
    #     @test all(!ismissing, samples[:x][z_samples])

    #     empirical_probs = zeros(length(partitions))
    #     w = map(x -> x.weight, samples.info[:samples])
    #     sum_weights = sum(w)
    #     z = z_samples

    #     for i in 1:size(z,1)
    #         partition = map(c -> findall(z[i,:,1] .== c), unique(z[i,:,1]))
    #         partition_idx = findfirst(p -> sort(p) == sort(partition), partitions)
    #         @test partition_idx !== nothing
    #         empirical_probs[partition_idx] += sum_weights == 0 ? 1 : w[i]
    #     end

    #     if sum_weights == 0
    #         empirical_probs /= length(w)
    #     end

    #     l2, discr = correct_posterior(empirical_probs, data, partitions, tau0, tau1, alpha, 1e-7)

    #     @test l2 < 0.05
    #     @test discr < 0.2
    # end
    # @testset "distributions" begin
    #     @testset "Representations" begin
    #         d = StickBreakingProcess(DirichletProcess(1.0))
    #         @test minimum(d) == 0
    #         @test maximum(d) == 1

    #         d = SizeBiasedSamplingProcess(DirichletProcess(1.0), 2.0)
    #         @test minimum(d) == 0
    #         @test maximum(d) == 2

    #         d = ChineseRestaurantProcess(DirichletProcess(1.0), [2, 1])
    #         @test minimum(d) == 1
    #         @test maximum(d) == 3
    #     end
    #     @testset "Dirichlet Process" begin

    #         α = 0.1
    #         N = 10_000

    #         # test SB representation
    #         d = StickBreakingProcess(DirichletProcess(α))
    #         Ev = mapreduce(_ -> rand(d), +, 1:N) / N
    #         @test Ev ≈ mean(Beta(1, α)) atol=0.05

    #         # test SBS representation
    #         d = SizeBiasedSamplingProcess(DirichletProcess(α), 2.0)
    #         Ej = mapreduce(_ -> rand(d), +, 1:N) / N
    #         @test Ej ≈ mean(Beta(1, α)) * 2 atol=0.05

    #         # test CRP representation
    #         d = ChineseRestaurantProcess(DirichletProcess(α), [2, 1])
    #         ks = map(_ -> rand(d), 1:N)
    #         c = map(k -> sum(ks .== k), support(d))
    #         p = c ./ sum(c)

    #         q = [2, 1, α] ./ (2 + α)
    #         q ./= sum(q)

    #         @test p[1] ≈ q[1] atol=0.1
    #         @test p[2] ≈ q[2] atol=0.1
    #         @test p[3] ≈ q[3] atol=0.1
    #     end
    #     @testset "Pitman-Yor Process" begin

    #         a = 0.5
    #         θ = 0.1
    #         t = 2
    #         N = 10_000

    #         # test SB representation
    #         d = StickBreakingProcess(PitmanYorProcess(a, θ, t))
    #         Ev = mapreduce(_ -> rand(d), +, 1:N) / N
    #         @test Ev ≈ mean(Beta(1 - a, θ + t*a)) atol=0.05

    #         # test SBS representation
    #         d = SizeBiasedSamplingProcess(PitmanYorProcess(a, θ, t), 2.0)
    #         Ej = mapreduce(_ -> rand(d), +, 1:N) / N
    #         @test Ej ≈ mean(Beta(1 - a, θ + t*a)) * 2 atol=0.05

    #         # test CRP representation
    #         d = ChineseRestaurantProcess(PitmanYorProcess(a, θ, t), [2, 1])
    #         ks = map(_ -> rand(d), 1:N)
    #         c = map(k -> sum(ks .== k), support(d))
    #         p = c ./ sum(c)

    #         q = [2 - a, 1 - a, θ + a*t] ./ (3 + θ)
    #         q ./= sum(q)

    #         @test p[1] ≈ q[1] atol=0.1
    #         @test p[2] ≈ q[2] atol=0.1
    #         @test p[3] ≈ q[3] atol=0.1
    #     end
    # end
    # @testset "stick breaking" begin
    #     # Data
    #     data = [-2,2,-1.5,1.5]

    #     # Base distribution
    #     mu_0 = mean(data)
    #     sigma_0 = 4
    #     sigma_1 = 0.5
    #     tau0 = 1/sigma_0^2
    #     tau1 = 1/sigma_1^2

    #     # DP parameters
    #     alpha = 0.25

    #     # stick-breaking process based on Papaspiliopoulos and Roberts (2008).
    #     @model function sbimm(y, rpm, trunc)
    #         # Base distribution.
    #         H = Normal(mu_0, sigma_0)

    #         # Latent assignments.
    #         N = length(y)
    #         z = tzeros(Int, N)

    #         # Infinite (truncated) collection of breaking points on unit stick.
    #         v = tzeros(Float64, trunc)

    #         # Cluster locations.
    #         x = tzeros(Float64, trunc)

    #         # Draw weights and locations.
    #         for k in 1:trunc
    #             v[k] ~ StickBreakingProcess(rpm)
    #             x[k] ~ H
    #         end

    #         # Weights.
    #         w = vcat(v[1], v[2:end] .* cumprod(1 .- v[1:end-1]))

    #         # Normalize weights to ensure they sum exactly to one.
    #         # This is required by the Categorical distribution in Distributions.
    #         w ./= sum(w)

    #         for i in 1:N
    #             # Draw location
    #             z[i] ~ Categorical(w)

    #             # Draw observation.
    #             y[i] ~ Normal(x[z[i]], sigma_1)
    #         end
    #     end

    #     rpm = DirichletProcess(alpha)

    #     sampler = SMC()
    #     mf = sbimm(data, rpm, 10)

    #     # Compute empirical posterior distribution over partitions
    #     samples = sample(mf, sampler, 10000)

    #     # Check that there is no NaN value associated
    #     z_samples = Int.(samples[:z])
    #     @test all(!isnan, samples[:x][z_samples])
    #     @test all(!ismissing, samples[:x][z_samples])

    #     empirical_probs = zeros(length(partitions))
    #     w = map(x -> x.weight, samples.info[:samples])
    #     sum_weights = sum(w)
    #     z = z_samples

    #     for i in 1:size(z,1)
    #         partition = map(c -> findall(z[i,:,1] .== c), unique(z[i,:,1]))
    #         partition_idx = findfirst(p -> sort(p) == sort(partition), partitions)
    #         @test partition_idx !== nothing
    #         empirical_probs[partition_idx] += sum_weights == 0 ? 1 : w[i]
    #     end

    #     if sum_weights == 0
    #         empirical_probs /= length(w)
    #     end

    #     l2, discr = correct_posterior(empirical_probs, data, partitions, tau0, tau1, alpha, 1e-7)

    #     # Increased ranges due to truncation error.
    #     @test l2 < 0.1
    #     @test discr < 0.3
    # end
    # @testset "size-based sampling" begin
    #     # Data
    #     data = [-2,2,-1.5,1.5]

    #     # Base distribution
    #     mu_0 = mean(data)
    #     sigma_0 = 4
    #     sigma_1 = 0.5
    #     tau0 = 1/sigma_0^2
    #     tau1 = 1/sigma_1^2

    #     # DP parameters
    #     alpha = 0.25

    #     # size-biased sampling process
    #     @model function sbsimm(y, rpm, trunc)
    #         # Base distribution.
    #         H = Normal(mu_0, sigma_0)

    #         # Latent assignments.
    #         N = length(y)
    #         z = tzeros(Int, N)

    #         x = tzeros(Float64, N)
    #         J = tzeros(Float64, N)

    #         k = 0
    #         surplus = 1.0

    #         for i in 1:N
    #             ps = vcat(J[1:k], surplus)
    #             z[i] ~ Categorical(ps)
    #             if z[i] > k
    #                 k = k + 1
    #                 J[k] ~ SizeBiasedSamplingProcess(rpm, surplus)
    #                 x[k] ~ H
    #                 surplus -= J[k]
    #             end
    #             y[i] ~ Normal(x[z[i]], sigma_1)
    #         end
    #     end

    #     rpm = DirichletProcess(alpha)

    #     sampler = SMC()
    #     mf = sbsimm(data, rpm, 100)

    #     # Compute empirical posterior distribution over partitions
    #     samples = sample(mf, sampler, 1000)

    #     # Check that there is no NaN value associated
    #     z_samples = Int.(samples[:z])
    #     @test all(!isnan, samples[:x][z_samples])
    #     @test all(!ismissing, samples[:x][z_samples])

    #     empirical_probs = zeros(length(partitions))
    #     w = map(x -> x.weight, samples.info[:samples])
    #     sum_weights = sum(w)
    #     z = z_samples

    #     for i in 1:size(z,1)
    #         partition = map(c -> findall(z[i,:,1] .== c), unique(z[i,:,1]))
    #         partition_idx = findfirst(p -> sort(p) == sort(partition), partitions)
    #         @test partition_idx !== nothing
    #         empirical_probs[partition_idx] += sum_weights == 0 ? 1 : w[i]
    #     end

    #     if sum_weights == 0
    #         empirical_probs /= length(w)
    #     end

    #     l2, discr = correct_posterior(empirical_probs, data, partitions, tau0, tau1, alpha, 1e-7)

    #     @test l2 < 0.05
    #     @test discr < 0.2
    # end
end

end
