module GibbsConditionalTests

using ..Models: gdemo, gdemo_default
using ..NumericalTests: check_gdemo, check_numerical
using ..ADUtils: adbackends
using Clustering: Clustering
using Distributions: Categorical, InverseGamma, Normal, sample
using ForwardDiff: ForwardDiff
using LinearAlgebra: Diagonal, I
using Random: Random
using ReverseDiff: ReverseDiff
using StableRNGs: StableRNG
using StatsBase: counts
using StatsFuns: StatsFuns
import Tapir
using Test: @test, @testset
using Turing

@testset "Testing gibbs conditionals.jl with $adbackend" for adbackend in adbackends
    Random.seed!(1000)
    rng = StableRNG(123)

    @testset "gdemo" begin
        # We consider the model
        # ```math
        # s ~ InverseGamma(2, 3)
        # m ~ Normal(0, √s)
        # xᵢ ~ Normal(m, √s), i = 1, …, N,
        # ```
        # with ``N = 2`` observations ``x₁ = 1.5`` and ``x₂ = 2``.

        # The conditionals and posterior can be formulated in terms of the following statistics:
        N = 2
        x_mean = 1.75 # sample mean ``∑ xᵢ / N``
        x_var = 0.0625 # sample variance ``∑ (xᵢ - x_bar)^2 / N``
        m_n = 3.5 / 3 # ``∑ xᵢ / (N + 1)``

        # Conditional distribution
        # ```math
        # m | s, x ~ Normal(m_n, sqrt(s / (N + 1)))
        # ```
        cond_m = let N = N, m_n = m_n
            c -> Normal(m_n, sqrt(c.s / (N + 1)))
        end

        # Conditional distribution
        # ```math
        # s | m, x ~ InverseGamma(2 + (N + 1) / 2, 3 + (m^2 + ∑ (xᵢ - m)^2) / 2) =
        #            InverseGamma(2 + (N + 1) / 2, 3 + m^2 / 2 + N / 2 * (x_var + (x_mean - m)^2))
        # ```
        cond_s = let N = N, x_mean = x_mean, x_var = x_var
            c -> InverseGamma(
                2 + (N + 1) / 2, 3 + c.m^2 / 2 + N / 2 * (x_var + (x_mean - c.m)^2)
            )
        end

        # Three Gibbs samplers:
        # one for each variable fixed to the posterior mean
        s_posterior_mean = 49 / 24
        sampler1 = Gibbs(
            GibbsConditional(:m, cond_m),
            GibbsConditional(:s, _ -> Normal(s_posterior_mean, 0)),
        )
        chain = sample(rng, gdemo_default, sampler1, 10_000)
        cond_m_mean = mean(cond_m((s=s_posterior_mean,)))
        check_numerical(chain, [:m, :s], [cond_m_mean, s_posterior_mean])
        @test all(==(s_posterior_mean), chain[:s][2:end])

        m_posterior_mean = 7 / 6
        sampler2 = Gibbs(
            GibbsConditional(:m, _ -> Normal(m_posterior_mean, 0)),
            GibbsConditional(:s, cond_s),
        )
        chain = sample(rng, gdemo_default, sampler2, 10_000)
        cond_s_mean = mean(cond_s((m=m_posterior_mean,)))
        check_numerical(chain, [:m, :s], [m_posterior_mean, cond_s_mean])
        @test all(==(m_posterior_mean), chain[:m][2:end])

        # and one for both using the conditional
        sampler3 = Gibbs(GibbsConditional(:m, cond_m), GibbsConditional(:s, cond_s))
        chain = sample(rng, gdemo_default, sampler3, 10_000)
        check_gdemo(chain)
    end

    @testset "GMM" begin
        Random.seed!(1000)
        rng = StableRNG(123)
        # We consider the model
        # ```math
        # μₖ ~ Normal(m, σ_μ), k = 1, …, K,
        # zᵢ ~ Categorical(π), i = 1, …, N,
        # xᵢ ~ Normal(μ_{zᵢ}, σₓ), i = 1, …, N,
        # ```
        # with ``K = 2`` clusters, ``N = 20`` observations, and the following parameters:
        K = 2 # number of clusters
        π = fill(1 / K, K) # uniform cluster weights
        m = 0.5 # prior mean of μₖ
        σ²_μ = 4.0 # prior variance of μₖ
        σ²_x = 0.01 # observation variance
        N = 20  # number of observations

        # We generate data
        μ_data = rand(rng, Normal(m, sqrt(σ²_μ)), K)
        z_data = rand(rng, Categorical(π), N)
        x_data = rand(rng, MvNormal(μ_data[z_data], σ²_x * I))

        @model function mixture(x)
            μ ~ $(MvNormal(fill(m, K), σ²_μ * I))
            z ~ $(filldist(Categorical(π), N))
            x ~ MvNormal(μ[z], $(σ²_x * I))
            return x
        end
        model = mixture(x_data)

        # Conditional distribution ``z | μ, x``
        # see http://www.cs.columbia.edu/~blei/fogm/2015F/notes/mixtures-and-gibbs.pdf
        cond_z = let x = x_data, log_π = log.(π), σ_x = sqrt(σ²_x)
            c -> begin
                dists = map(x) do xi
                    logp = log_π .+ logpdf.(Normal.(c.μ, σ_x), xi)
                    return Categorical(StatsFuns.softmax!(logp))
                end
                return arraydist(dists)
            end
        end

        # Conditional distribution ``μ | z, x``
        # see http://www.cs.columbia.edu/~blei/fogm/2015F/notes/mixtures-and-gibbs.pdf
        cond_μ = let K = K, x_data = x_data, inv_σ²_μ = inv(σ²_μ), inv_σ²_x = inv(σ²_x)
            c -> begin
                # Convert cluster assignments to one-hot encodings
                z_onehot = c.z .== (1:K)'

                # Count number of observations in each cluster
                n = vec(sum(z_onehot; dims=1))

                # Compute mean and variance of the conditional distribution
                μ_var = @. inv(inv_σ²_x * n + inv_σ²_μ)
                μ_mean = (z_onehot' * x_data) .* inv_σ²_x .* μ_var

                return MvNormal(μ_mean, Diagonal(μ_var))
            end
        end

        estimate(chain, var) = dropdims(mean(Array(group(chain, var)); dims=1); dims=1)
        function estimatez(chain, var, range)
            z = Int.(Array(group(chain, var)))
            return map(i -> findmax(counts(z[:, i], range))[2], 1:size(z, 2))
        end

        lμ_data, uμ_data = extrema(μ_data)

        # Compare three Gibbs samplers
        sampler1 = Gibbs(GibbsConditional(:z, cond_z), GibbsConditional(:μ, cond_μ))
        sampler2 = Gibbs(GibbsConditional(:z, cond_z), MH(:μ))
        sampler3 = Gibbs(GibbsConditional(:z, cond_z), HMC(0.01, 7, :μ; adtype=adbackend))
        for sampler in (sampler1, sampler2, sampler3)
            chain = sample(rng, model, sampler, 10_000)

            μ_hat = estimate(chain, :μ)
            lμ_hat, uμ_hat = extrema(μ_hat)
            @test isapprox([lμ_data, uμ_data], [lμ_hat, uμ_hat], atol=0.1)

            z_hat = estimatez(chain, :z, 1:2)
            ari, _, _, _ = Clustering.randindex(z_data, Int.(z_hat))
            @test isapprox(ari, 1, atol=0.1)
        end
    end
end

end
