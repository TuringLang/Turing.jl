@model sv_model(y) = begin
  T = length(y)
  ϕ ~ Uniform(-1, 1)
  σ ~ Truncated(Cauchy(0,5), 0, +Inf)
  μ ~ Cauchy(0, 10)
  h = tzeros(Real, T)
  h[1] ~ Normal(μ, σ / sqrt(1 - ϕ^2))
  y[1] ~ Normal(0, exp.(h[1] / 2))
  for t = 2:T
    h[t] ~ Normal(μ + ϕ * (h[t-1] - μ) , σ)
    y[t] ~ Normal(0, exp.(h[t] / 2))
  end
end
