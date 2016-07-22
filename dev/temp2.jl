# Importance Sampling
# Task: estimate <ϕ(x)> = ∫P(x)ϕ(x)dx

# Helper Functions
function nor(x::Real, μ::Real, σ::Real)
  return 1 / (sqrt(2 * pi) * σ) * exp( (x - μ)^2 / (-2 * σ^2))
end

nor(0, 0, 1)

# The Real Distribution
function P(x::Real)
  return 0.8 * nor(x, 0, 0.25) + 0.2 * nor(x, 1, 0.25)
end

# The Estimate Distribution
function Q(x::Real)
  return nor(x, 0, 0.25)
end

# The function
function ϕ(x::Real)
  return 0.1x
end

# IS
N = 50
samples = [randn() for _ in 1:N]
samples
weights = [P(samples[i]) / Q(samples[i]) for i in 1:N]
μ̂ = sum(weights .* map(ϕ, samples)) / sum(weights)

# Plot P and Q
using Gadfly
function_layer = layer([P, Q, ϕ], -5, 5)
sample_layer = layer(x=samples, y=zeros(Float64, N), Geom.point)
weight_layer = layer(x=samples, y=weights, Geom.point, Theme(default_color=colorant"green"))
IS_plot = plot(function_layer, sample_layer, weight_layer, Guide.xlabel("x"), Guide.ylabel("f(x)"), Guide.title("Importance Sampling"))
draw(PNG("IS.png", 7inch, 7inch), IS_plot)
# using a normal distribution f2() to draw samples to estimate ∫f1(x)f3(x)

vline_layer = layer(xintercept=[0.8, 1.6], Geom.vline, Theme(default_color=colorant"green"))
IS_plot = plot(function_layer, vline_layer, Guide.xlabel("x"), Guide.ylabel("f(x)"), Guide.title("Bad Q(x)"))
draw(PNG("IS2.png", 7inch, 7inch), IS_plot)
