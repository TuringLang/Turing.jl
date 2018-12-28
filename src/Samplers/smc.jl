"""
    SMC(n_particles::Int)

Sequential Monte Carlo sampler.

Note that this method is particle-based, and arrays of variables
must be stored in a [`TArray`](@ref) object.

Usage:

```julia
SMC(1000)
```

Example:

```julia
# Define a simple Normal model with unknown mean and variance.
@model gdemo(x) = begin
  s ~ InverseGamma(2,3)
  m ~ Normal(0, sqrt(s))
  x[1] ~ Normal(m, sqrt(s))
  x[2] ~ Normal(m, sqrt(s))
  return s, m
end

sample(gdemo([1.5, 2]), SMC(1000))
```
"""
mutable struct SMC{space, F} <: InferenceAlgorithm
    n_particles           ::  Int
    resampler             ::  F
    resampler_threshold   ::  Float64
    gid                   ::  Int
end
SMC(n) = SMC(n, resample_systematic, 0.5, 0)
function SMC(n_particles::Int, space...)
    F = typeof(resample_systematic)
    SMC{space, F}(n_particles, resample_systematic, 0.5, 0)
end
function SMC(alg::SMC{space, F}, new_gid::Int) where {space, F}
    SMC{space, F}(alg.n_particles, alg.resampler, alg.resampler_threshold, new_gid)
end

getspace(::SMC{space}) where space = space

function Sampler(alg::SMC)
    info = Dict{Symbol, Any}()
    info[:logevidence] = []
    Sampler(alg, info)
end
