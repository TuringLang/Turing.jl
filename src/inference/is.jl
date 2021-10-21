"""
    IS()

Importance sampling algorithm.

Note that this method is particle-based, and arrays of variables
must be stored in a [`TArray`](@ref) object.

Usage:

```julia
IS()
```

Example:

```julia
# Define a simple Normal model with unknown mean and variance.
@model gdemo(x) = begin
    s² ~ InverseGamma(2,3)
    m ~ Normal(0,sqrt.(s))
    x[1] ~ Normal(m, sqrt.(s))
    x[2] ~ Normal(m, sqrt.(s))
    return s², m
end

sample(gdemo([1.5, 2]), IS(), 1000)
```
"""
struct IS{space} <: InferenceAlgorithm end

IS() = IS{()}()

DynamicPPL.initialsampler(sampler::Sampler{<:IS}) = sampler

function DynamicPPL.initialstep(
    rng::AbstractRNG,
    model::Model,
    spl::Sampler{<:IS},
    vi::AbstractVarInfo;
    kwargs...
)
    return Transition(vi), nothing
end

function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::Model,
    spl::Sampler{<:IS},
    ::Nothing;
    kwargs...
)
    vi = VarInfo(rng, model, spl)
    return Transition(vi), nothing
end

# Calculate evidence.
function getlogevidence(samples::Vector{<:Transition}, ::Sampler{<:IS}, state)
    return logsumexp(map(x -> x.lp, samples)) - log(length(samples))
end

function DynamicPPL.assume(rng, spl::Sampler{<:IS}, dist::Distribution, vn::VarName, vi)
    if haskey(vi, vn)
        r = vi[vn]
    else
        r = rand(rng, dist)
        push!(vi, vn, r, dist, spl)
    end
    return r, 0
end

function DynamicPPL.observe(spl::Sampler{<:IS}, dist::Distribution, value, vi)
    return logpdf(dist, value)
end
