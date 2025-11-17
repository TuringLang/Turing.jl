module GibbsConditionalTests

using Distributions: InverseGamma, Normal
using Distributions: sample
using DynamicPPL: DynamicPPL
using Random: Random
using StableRNGs: StableRNG
using Test: @inferred, @test, @test_broken, @test_throws, @testset
using Turing

@testset "GibbsConditional" begin
    @model function inverse_gdemo(x)
        λ ~ Gamma(2, inv(3))
        m ~ Normal(0, sqrt(1 / λ))
        for i in 1:length(x)
            x[i] ~ Normal(m, sqrt(1 / λ))
        end
    end

    # Define analytical conditionals
    function cond_λ(c)
        a = 2.0
        b = inv(3)
        m = c[@varname(m)]
        x = c[@varname(x)]
        n = length(x)
        a_new = a + (n + 1) / 2
        b_new = b + sum((x[i] - m)^2 for i in 1:n) / 2 + m^2 / 2
        return Gamma(a_new, 1 / b_new)
    end

    function cond_m(c)
        λ = c[@varname(λ)]
        x = c[@varname(x)]
        n = length(x)
        m_mean = sum(x) / (n + 1)
        m_var = 1 / (λ * (n + 1))
        return Normal(m_mean, sqrt(m_var))
    end

    # Test basic functionality
    @testset "basic sampling" begin
        Random.seed!(42)
        x_obs = [1.0, 2.0, 3.0, 2.5, 1.5]
        model = inverse_gdemo(x_obs)

        # Test that GibbsConditional works
        sampler = Gibbs(:λ => GibbsConditional(cond_λ), :m => GibbsConditional(cond_m))
        chain = sample(model, sampler, 1000)

        # Check that we got the expected variables
        @test :λ in names(chain)
        @test :m in names(chain)

        # Check that the values are reasonable
        λ_samples = vec(chain[:λ])
        m_samples = vec(chain[:m])

        # Given the observed data, we expect certain behavior
        @test mean(λ_samples) > 0  # λ should be positive
        @test minimum(λ_samples) > 0
        @test std(m_samples) < 2.0  # m should be relatively well-constrained
    end

    # Test mixing with other samplers
    @testset "mixed samplers" begin
        x_obs = [1.0, 2.0, 3.0]
        model = inverse_gdemo(x_obs)

        # Mix GibbsConditional with standard samplers
        sampler = Gibbs(:λ => GibbsConditional(cond_λ), :m => MH())
        chain = sample(model, sampler, 500)

        @test :λ in names(chain)
        @test :m in names(chain)
        @test size(chain, 1) == 500
    end

    # Test with a simpler model
    @testset "simple normal model" begin
        @model function simple_normal(dim)
            μ ~ Normal(0, 10)
            σ2 ~ truncated(Normal(1, 1); lower=0.01)
            return x ~ MvNormal(fill(μ, dim), I * σ2)
        end

        # Conditional for μ given σ and x
        function cond_μ(c)
            σ2 = c[@varname(σ2)]
            x = c[@varname(x)]
            n = length(x)
            # Prior: μ ~ Normal(0, 10)
            # Likelihood: x[i] ~ Normal(μ, σ)
            # Posterior: μ ~ Normal(μ_post, σ_post)
            prior_var = 100.0  # 10^2
            post_var = 1 / (1 / prior_var + n / σ2)
            post_mean = post_var * (0 / prior_var + sum(x) / σ2)
            return Normal(post_mean, sqrt(post_var))
        end

        rng = StableRNG(23)
        dim = 10_000
        true_mean = 2.0
        x_obs = randn(rng, dim) .+ true_mean
        model = simple_normal(dim) | (; x=x_obs)
        sampler = Gibbs(:μ => GibbsConditional(cond_μ), :σ2 => MH())
        chain = sample(rng, model, sampler, 1_000)
        # The correct posterior mean isn't true_mean, but it is very close, because we
        # have a lot of data.
        @test mean(chain, :μ) ≈ true_mean atol = 0.02
    end

    # Test that GibbsConditional is marked as a valid component
    @testset "isgibbscomponent" begin
        gc = GibbsConditional(c -> Normal(0, 1))
        @test Turing.Inference.isgibbscomponent(gc)
    end
end

end
