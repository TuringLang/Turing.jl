"""
    IS()

Importance sampling algorithm.

Usage:

```julia
IS()
```

Example:

```julia
# Define a simple Normal model with unknown mean and variance.
@model function gdemo(x)
    s² ~ InverseGamma(2,3)
    m ~ Normal(0,sqrt.(s))
    x[1] ~ Normal(m, sqrt.(s))
    x[2] ~ Normal(m, sqrt.(s))
    return s², m
end

sample(gdemo([1.5, 2]), IS(), 1000)
```
"""
struct IS <: InferenceAlgorithm end

DynamicPPL.initialsampler(sampler::Sampler{<:IS}) = sampler

function DynamicPPL.initialstep(
    rng::AbstractRNG, model::Model, spl::Sampler{<:IS}, vi::AbstractVarInfo; kwargs...
)
    return Transition(model, vi), nothing
end

function AbstractMCMC.step(
    rng::Random.AbstractRNG, model::Model, spl::Sampler{<:IS}, ::Nothing; kwargs...
)
    vi = VarInfo(rng, model, spl)
    return Transition(model, vi), nothing
end

# Calculate evidence.
function getlogevidence(samples::Vector{<:Transition}, ::Sampler{<:IS}, state)
    return logsumexp(map(x -> x.lp, samples)) - log(length(samples))
end

function DynamicPPL.assume(rng, ::Sampler{<:IS}, dist::Distribution, vn::VarName, vi)
    if haskey(vi, vn)
        r = vi[vn]
    else
        r = rand(rng, dist)
        vi = push!!(vi, vn, r, dist)
    end
    return r, 0, vi
end
