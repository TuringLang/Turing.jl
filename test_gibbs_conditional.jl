using Turing
using Turing.Inference: GibbsConditional
using Distributions
using Random
using Statistics
using Test

# Test with the inverse gamma example from the issue
@model function inverse_gdemo(x)
    λ ~ Gamma(2, inv(3))
    m ~ Normal(0, sqrt(1 / λ))
    for i in 1:length(x)
        x[i] ~ Normal(m, sqrt(1 / λ))
    end
end

# Define analytical conditionals
function cond_λ(c::NamedTuple)
    a = 2.0
    b = inv(3)
    m = c.m
    x = c.x
    n = length(x)
    a_new = a + (n + 1) / 2
    b_new = b + sum((x[i] - m)^2 for i in 1:n) / 2 + m^2 / 2
    return Gamma(a_new, 1 / b_new)
end

function cond_m(c::NamedTuple)
    λ = c.λ
    x = c.x
    n = length(x)
    m_mean = sum(x) / (n + 1)
    m_var = 1 / (λ * (n + 1))
    return Normal(m_mean, sqrt(m_var))
end

@testset "GibbsConditional Integration Tests" begin
    # Generate some observed data
    Random.seed!(42)
    x_obs = [1.0, 2.0, 3.0, 2.5, 1.5]

    # Create the model
    model = inverse_gdemo(x_obs)

    @testset "Basic GibbsConditional sampling" begin
        # Sample using GibbsConditional
        sampler = Gibbs(:λ => GibbsConditional(:λ, cond_λ), :m => GibbsConditional(:m, cond_m))

        # Run a short chain to test
        chain = sample(model, sampler, 100)

        # Test that sampling completed successfully
        @test chain isa MCMCChains.Chains
        @test size(chain, 1) == 100
        @test :λ in names(chain)
        @test :m in names(chain)
    end

    @testset "Sample statistics" begin
        # Generate samples for statistics testing
        sampler = Gibbs(:λ => GibbsConditional(:λ, cond_λ), :m => GibbsConditional(:m, cond_m))
        chain = sample(model, sampler, 100)

        # Extract samples
        λ_samples = vec(chain[:λ])
        m_samples = vec(chain[:m])

        # Test λ statistics
        @test mean(λ_samples) > 0  # λ should be positive
        @test minimum(λ_samples) > 0  # All λ samples should be positive
        @test std(λ_samples) > 0  # Should have some variability
        @test isfinite(mean(λ_samples))
        @test isfinite(std(λ_samples))

        # Test m statistics
        @test isfinite(mean(m_samples))
        @test isfinite(std(m_samples))
        @test std(m_samples) > 0  # Should have some variability
    end

    @testset "Mixed samplers" begin
        # Test mixing with other samplers
        sampler2 = Gibbs(:λ => GibbsConditional(:λ, cond_λ), :m => MH())

        chain2 = sample(model, sampler2, 100)

        # Test that mixed sampling completed successfully
        @test chain2 isa MCMCChains.Chains
        @test size(chain2, 1) == 100
        @test :λ in names(chain2)
        @test :m in names(chain2)

        # Test that values are reasonable
        λ_samples2 = vec(chain2[:λ])
        m_samples2 = vec(chain2[:m])
        @test all(λ_samples2 .> 0)  # All λ should be positive
        @test all(isfinite.(λ_samples2))
        @test all(isfinite.(m_samples2))
    end
end
