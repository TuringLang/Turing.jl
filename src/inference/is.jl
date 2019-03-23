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

function init_samples(alg::IS, sample::Tsample; kwargs...) where {Tsample <: Sample}
    return Array{Tsample}(undef, alg.n_particles)
end

function init_spl(model, alg::IS; stable = true, kwargs...)
    vi = VarInfo(model)
    spl = Sampler(alg, nothing)
    return spl, vi
end

function _sample(args...; stable = true, kwargs...)
    if stable
        return _sample_stable(args...)
    else
        return _sample_unstable(args...)
    end
end

function _sample_stable(vi, samples, spl, model, alg::IS)
    n = spl.alg.n_particles
    for i = 1:n
        vi = empty!(deepcopy(vi))
        model(vi, spl)
        samples[i] = Sample(vi)
    end

    le = logsumexp(map(x->x[:lp], samples)) - log(n)

    return Chain(le, samples)
end

function _sample_unstable(vi, samples, spl, model, alg::IS)
    n = spl.alg.n_particles
    for i = 1:n
        vi = VarInfo()
        model(vi, spl)
        samples[i] = Sample(vi)
    end

    le = logsumexp(map(x->x[:lp], samples)) - log(n)

    return Chain(le, samples)
end

function assume(spl::Sampler{<:IS}, dist::Distribution, vn::VarName, vi::AbstractVarInfo)
    r = rand(dist)
    push!(vi, vn, r, dist, 0)
    return r, 0.0
end

function observe(spl::Sampler{<:IS}, dist::Distribution, value::Any, vi::AbstractVarInfo)
    # acclogp!(vi, logpdf(dist, value))
    return logpdf(dist, value)
end
