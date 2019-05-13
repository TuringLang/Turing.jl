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
mutable struct IS <: InferenceAlgorithm
    n_particles ::  Int
end

function Sampler(alg::IS, s::Selector)
    info = Dict{Symbol, Any}()
    Sampler(alg, info, s)
end

function sample(model::Model, alg::IS)
    spl = Sampler(alg);
    samples = Array{Sample}(undef, alg.n_particles)

    n = spl.alg.n_particles
    for i = 1:n
        vi = VarInfo()
        model(vi, spl)
        samples[i] = Sample(vi)
    end

    le = logsumexp(map(x->x[:lp], samples)) - log(n)

    Chain(le, samples)
end

function assume(spl::Sampler{<:IS}, dist::Distribution, vn::VarName, vi::VarInfo)
    r = rand(dist)
    push!(vi, vn, r, dist, spl)
    r, zero(Real)
end

function observe(spl::Sampler{<:IS}, dist::Distribution, value::Any, vi::VarInfo)
    # acclogp!(vi, logpdf(dist, value))
    logpdf(dist, value)
end
