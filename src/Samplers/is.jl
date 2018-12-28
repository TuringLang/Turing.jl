"""
    IS(n_particles::Int)

Importance sampling algorithm.

Note that this method is particle-based, and arrays of variables
must be stored in a [`TArray`](@ref) object.

Arguments:

- `n_particles` is the number of particles to use.

Usage:

```julia
IS(1000)
```

Example:

```julia
# Define a simple Normal model with unknown mean and variance.
@model gdemo(x) = begin
    s ~ InverseGamma(2,3)
    m ~ Normal(0,sqrt.(s))
    x[1] ~ Normal(m, sqrt.(s))
    x[2] ~ Normal(m, sqrt.(s))
    return s, m
end

sample(gdemo([1.5, 2]), IS(1000))
```
"""
mutable struct IS{space} <: InferenceAlgorithm
    n_particles ::  Int
end
IS(n_particles) = IS{()}(n_particles)

function Sampler(alg::IS)
    info = Dict{Symbol, Any}()
    Sampler(alg, info)
end
