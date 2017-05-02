
doc"""
    IS(n_particles::Int)

Importance sampling algorithm object.

- `n_particles` is the number of particles to use

Usage:

```julia
IS(1000)
```

Example:

```julia
# Define a simple Normal model with unknown mean and variance.
@model gdemo(x) = begin
  s ~ InverseGamma(2,3)
  m ~ Normal(0,sqrt(s))
  x[1] ~ Normal(m, sqrt(s))
  x[2] ~ Normal(m, sqrt(s))
  return s, m
end

sample(gdemo([1.5, 2]), IS(1000))
```
"""
immutable IS <: InferenceAlgorithm
  n_particles ::  Int
end

Sampler(alg::IS) = begin
  info = Dict{Symbol, Any}()
  Sampler(alg, info)
end

sample(model::Function, alg::IS) = begin
  spl = Sampler(alg);
  samples = Array{Sample}(alg.n_particles)

  n = spl.alg.n_particles
  for i = 1:n
    vi = model(vi=VarInfo(), sampler=spl)
    samples[i] = Sample(vi)
  end

  le = logsum(map(x->x[:lp], samples)) - log(n)

  Chain(exp(le), samples)
end

assume(spl::Sampler{IS}, dist::Distribution, vn::VarName, vi::VarInfo) = begin
  r = rand(dist)
  push!(vi, vn, vectorize(dist, r), dist, 0)
  r
end
observe(spl::Sampler{IS}, dist::Distribution, value, vi::VarInfo) = vi.logp += logpdf(dist, value)
