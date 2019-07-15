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
transition_type(::Sampler{<:IS}) = Transition
alg_str(::Sampler{<:IS}) = "IS"

mutable struct ISState <: SamplerState
    vi                 ::  TypedVarInfo
    final_logevidence  ::  Float64
end

ISState(model::Model) = ISState(VarInfo(model), 0.0)

function Sampler(alg::IS, model::Model, s::Selector)
    info = Dict{Symbol, Any}()
    state = ISState(model)
    Sampler(alg, info, s, state)
end

function step!(
    ::AbstractRNG,
    model::Model,
    spl::Sampler{<:IS},
    ::Integer;
    kwargs...
)
    empty!(spl.state.vi)
    model(spl.state.vi, spl)

    return transition(spl)
end

function sample(model::Model, alg::IS)
    spl = Sampler(alg);
    samples = Array{Sample}(undef, alg.n_particles)

    n = spl.alg.n_particles
    vi = VarInfo(model)
    for i = 1:n
        empty!(vi)
        model(vi, spl)
        samples[i] = Sample(vi)
    end


    Chain(le, samples)
end

function sample_end!(
    ::AbstractRNG,
    ::Model,
    spl::Sampler{<:IS},
    N::Integer,
    ts::Vector{Transition};
    kwargs...
) where {SamplerType<:AbstractSampler}
    # Calculate evidence.
    spl.state.final_logevidence = logsumexp(map(x->x.lp, ts)) - log(N)
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
