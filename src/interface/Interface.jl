module Interface

import Distributions: sample, Sampleable
import Random: GLOBAL_RNG, AbstractRNG

export AbstractSampler,
       AbstractTransition,
       sample_init!,
       sample_end!,
       sample,
       step!

"""
    AbstractSampler

The `AbstractSampler` type is intended to be inherited from when
implementing a custom sampler. Any persistent state information should be
saved in a subtype of `AbstractSampler`.

When defining a new sampler, you should also overload the function
`transition_type`, which tells the `sample` function what type of parameter
it should expect to receive.
"""
abstract type AbstractSampler end

"""
    AbstractTransition

The `AbstractTransition` type describes the results of a single step
of a given sampler. As an example, one implementation of an
`AbstractTransition` might include be a vector of parameters sampled from
a prior distribution.

Transition types should store a single draw from any sampler, since the
interface will sample `N` times, and store the results of each step in an
array of type `Array{Transition<:AbstractTransition, 1}`. If you were
using a sampler that returned a `NamedTuple` after each step, your
implementation might look like:

```
struct MyTransition <: AbstractTransition
    draw :: NamedTuple
end
```
"""
abstract type AbstractTransition end


"""
    sample(
        ℓ::ModelType,
        s::SamplerType,
        N::Integer;
        kwargs...)

    sample(
        rng::AbstractRNG,
        ℓ::ModelType,
        s::SamplerType,
        N::Integer;
        kwargs...)

A generic interface for samplers.
"""
function sample(
    ℓ::ModelType,
    s::SamplerType,
    N::Integer;
    kwargs...
) where {ModelType<:Sampleable, SamplerType<:AbstractSampler}
    return sample(GLOBAL_RNG, ℓ, s, N)
end

"""
    sample(
        rng::AbstractRNG,
        ℓ::ModelType,
        s::SamplerType,
        N::Integer;
        kwargs...
    )

`sample` returns an `MCMCChains.Chains` object containing `N` samples from a given model and
sampler.
"""
function sample(
    rng::AbstractRNG,
    ℓ::ModelType,
    s::SamplerType,
    N::Integer;
    kwargs...
) where {ModelType<:Sampleable, SamplerType<:AbstractSampler}
    # Perform any necessary setup.
    sample_init!(rng, ℓ, s, N; kwargs...)

    # Preallocate the TransitionType vector.
    ts = transitions_init(rng, ℓ, s, N; kwargs...)

    # Step through the sampler.
    for i=1:N
        if i == 1
            ts[i] = step!(rng, ℓ, s, N; kwargs...)
        else
            ts[i] = step!(rng, ℓ, s, N, ts[i-1]; kwargs...)
        end
    end

    # Wrap up the sampler, if necessary.
    sample_end!(rng, ℓ, s, N, ts; kwargs...)

    return Chains(ts)
end

"""
    sample_init!(
        rng::AbstractRNG,
        ℓ::ModelType,
        s::SamplerType,
        N::Integer;
        kwargs...
    )

Performs whatever initial setup is required for your sampler.
"""
function sample_init!(
    rng::AbstractRNG,
    ℓ::ModelType,
    s::SamplerType,
    N::Integer;
    kwargs...
) where {ModelType<:Sampleable, SamplerType<:AbstractSampler}
    # Do nothing.
    @warn "No sample_init! function has been implemented for objects
           of types $(typeof(ℓ)) and $(typeof(s))"
end

"""
    sample_end!(
        rng::AbstractRNG,
        ℓ::ModelType,
        s::SamplerType,
        N::Integer,
        ts::Vector{TransitionType};
        kwargs...
    )

Performs whatever finalizing the sampler requires.
"""
function sample_end!(
    rng::AbstractRNG,
    ℓ::ModelType,
    s::SamplerType,
    N::Integer,
    ts::Vector{TransitionType};
    kwargs...
) where {
    ModelType<:Sampleable,
    SamplerType<:AbstractSampler,
    TransitionType<:AbstractTransition
}
    # Do nothing.
    @warn "No sample_end! function has been implemented for objects
           of types $(typeof(ℓ)) and $(typeof(s))"
end

"""
    step!(
        rng::AbstractRNG,
        ℓ::ModelType,
        s::SamplerType,
        N::Integer;
        kwargs...
    )

Returns a single `AbstractTransition` drawn using the model and sampler type.
This is a unique step function called the first time a sampler runs.
"""
function step!(
    rng::AbstractRNG,
    ℓ::ModelType,
    s::SamplerType,
    N::Integer;
    kwargs...
) where {ModelType<:Sampleable, SamplerType<:AbstractSampler}
    # Do nothing.
    @warn "No step! function has been implemented for objects
           of types $(typeof(ℓ)) and $(typeof(s))"
end

"""
    step!(
        rng::AbstractRNG,
        ℓ::ModelType,
        s::SamplerType,
        N::Integer,
        t::TransitionType;
        kwargs...
    )

Returns a single `AbstractTransition` drawn using the model and sampler type.
"""
function step!(
    rng::AbstractRNG,
    ℓ::ModelType,
    s::SamplerType,
    N::Integer,
    t::TransitionType;
    kwargs...
) where {ModelType<:Sampleable,
    SamplerType<:AbstractSampler,
    TransitionType<:AbstractTransition
}
    # Do nothing.
    @warn "No step! function has been implemented for objects
           of types $(typeof(ℓ)) and $(typeof(s))"
end

"""
    transitions_init(
        rng::AbstractRNG,
        ℓ::ModelType,
        s::SamplerType,
        N::Integer;
        kwargs...
    )

Generates a vector of `AbstractTransition` types of length `N`.
"""
function transitions_init(
    rng::AbstractRNG,
    ℓ::ModelType,
    s::SamplerType,
    N::Integer;
    kwargs...
) where {ModelType<:Sampleable, SamplerType<:AbstractSampler}
    @warn "No transitions_init function has been implemented
           for objects of types $(typeof(ℓ)) and $(typeof(s))"
    return Vector(undef, N)
end

end # module Interface
