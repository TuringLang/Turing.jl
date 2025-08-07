using Turing
using Turing.Inference: GibbsConditional
using Distributions
using Random
using Statistics

# Test with the inverse gamma example from the issue
@model function inverse_gdemo(x)
    λ ~ Gamma(2, 3)
    m ~ Normal(0, sqrt(1 / λ))
    for i in 1:length(x)
        x[i] ~ Normal(m, sqrt(1 / λ))
    end
end

# Define analytical conditionals
function cond_λ(c::NamedTuple)
    a = 2.0
    b = 3.0
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

# Generate some observed data
Random.seed!(42)
x_obs = [1.0, 2.0, 3.0, 2.5, 1.5]

# Create the model
model = inverse_gdemo(x_obs)

# Sample using GibbsConditional
println("Testing GibbsConditional sampler...")
sampler = Gibbs(:λ => GibbsConditional(:λ, cond_λ), :m => GibbsConditional(:m, cond_m))

# Run a short chain to test
chain = sample(model, sampler, 100)

println("Sampling completed successfully!")
println("\nChain summary:")
println(chain)

# Extract samples
λ_samples = vec(chain[:λ])
m_samples = vec(chain[:m])

println("\nλ statistics:")
println("  Mean: ", mean(λ_samples))
println("  Std:  ", std(λ_samples))
println("  Min:  ", minimum(λ_samples))
println("  Max:  ", maximum(λ_samples))

println("\nm statistics:")
println("  Mean: ", mean(m_samples))
println("  Std:  ", std(m_samples))
println("  Min:  ", minimum(m_samples))
println("  Max:  ", maximum(m_samples))

# Test mixing with other samplers
println("\n\nTesting mixed samplers...")
sampler2 = Gibbs(:λ => GibbsConditional(:λ, cond_λ), :m => MH())

chain2 = sample(model, sampler2, 100)
println("Mixed sampling completed successfully!")
println("\nMixed chain summary:")
println(chain2)
