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
    s ~ InverseGamma(2,3)
    m ~ Normal(0,sqrt.(s))
    x[1] ~ Normal(m, sqrt.(s))
    x[2] ~ Normal(m, sqrt.(s))
    return s, m
end

sample(gdemo([1.5, 2]), IS(), 1000)
```
"""
struct IS{space} <: InferenceAlgorithm end

IS() = IS{()}()

mutable struct ISState{V<:VarInfo, F<:AbstractFloat} <: AbstractSamplerState
    vi                 ::  V
    final_logevidence  ::  F
end

ISState(model::Model) = ISState(VarInfo(model), 0.0)

function Sampler(alg::IS, model::Model, s::Selector)
    info = Dict{Symbol, Any}()
    state = ISState(model)
    return Sampler(alg, info, s, state)
end

function AbstractMCMC.step!(
    rng::AbstractRNG,
    model::Model,
    spl::Sampler{<:IS},
    ::Integer,
    transition;
    kwargs...
)
    empty!(spl.state.vi)
    model(rng, spl.state.vi, spl)

    return Transition(spl)
end

function AbstractMCMC.sample_end!(
    ::AbstractRNG,
    ::Model,
    spl::Sampler{<:IS},
    N::Integer,
    ts::Vector;
    kwargs...
)
    # Calculate evidence.
    spl.state.final_logevidence = logsumexp(map(x->x.lp, ts)) - log(N)
end

function DynamicPPL.assume(rng, spl::Sampler{<:IS}, dist::Distribution, vn::VarName, vi)
    r = rand(rng, dist)
    push!(vi, vn, r, dist, spl)
    return r, 0
end

function DynamicPPL.observe(spl::Sampler{<:IS}, dist::Distribution, value, vi)
    # acclogp!(vi, logpdf(dist, value))
    return logpdf(dist, value)
end
