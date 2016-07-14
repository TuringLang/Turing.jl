

# Mapping
function map_distr(d :: Distribution)
  if typeof(d) == Distributions.Normal
    return hmcNormal
  elseif typeof(d) == Distributions.InverseGamma
    return hmcInverseGamma
  else
    error("[assume] unsupported distribution: $(typeof(d))")
  end
end

# InverseGamma
function hmcInverseGamma(s, c)
  return v -> 1 / (gamma(c) / s) * (1 / (s * v))^(c + 1) * exp(-1 / (s * v))
end

# Normal
function hmcNormal(μ, σ)
  return x::Real -> 1 / sqrt((2pi)^2 * σ^2) * exp(-0.5 * (x - μ)^2 / σ^2)
end


a = Normal(1, 2)
map_distr(a)

parse(a)
