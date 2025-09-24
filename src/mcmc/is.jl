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

function DynamicPPL.initialstep(
    rng::AbstractRNG, model::Model, spl::Sampler{<:IS}, vi::AbstractVarInfo; kwargs...
)
    return Transition(model, vi, nothing), nothing
end

function AbstractMCMC.step(
    rng::Random.AbstractRNG, model::Model, spl::Sampler{<:IS}, ::Nothing; kwargs...
)
    model = DynamicPPL.setleafcontext(model, ISContext(rng))
    _, vi = DynamicPPL.evaluate!!(model, DynamicPPL.VarInfo())
    vi = DynamicPPL.typed_varinfo(vi, model)
    return Transition(model, vi, nothing), nothing
end

# Calculate evidence.
function getlogevidence(samples::Vector{<:Transition}, ::Sampler{<:IS}, state)
    return logsumexp(map(x -> x.loglikelihood, samples)) - log(length(samples))
end

struct ISContext{R<:AbstractRNG} <: DynamicPPL.AbstractContext
    rng::R
end
DynamicPPL.NodeTrait(::ISContext) = DynamicPPL.IsLeaf()

function DynamicPPL.tilde_assume!!(
    ctx::ISContext, dist::Distribution, vn::VarName, vi::AbstractVarInfo
)
    if haskey(vi, vn)
        r = vi[vn]
    else
        r = rand(ctx.rng, dist)
        vi = push!!(vi, vn, r, dist)
    end
    vi = DynamicPPL.accumulate_assume!!(vi, r, 0.0, vn, dist)
    return r, vi
end
function DynamicPPL.tilde_observe!!(
    ::ISContext, right::Distribution, left, vn::Union{VarName,Nothing}, vi::AbstractVarInfo
)
    return DynamicPPL.tilde_observe!!(DefaultContext(), right, left, vn, vi)
end
