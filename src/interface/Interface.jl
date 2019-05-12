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

transition_type(s::AbstractSampler) = AbstractTransition

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
    t = Array{transition_type(s), 1}(undef, N)

    # Step through the sampler.
    for i=1:N
        t[i] = step!(rng, ℓ, s, N; kwargs...)
    end

    # Wrap up the sampler, if necessary.
    sample_end!(rng, ℓ, s, N; kwargs...)

    return Chains(t)
end

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

function sample_end!(
    rng::AbstractRNG,
    ℓ::ModelType,
    s::SamplerType,
    N::Integer;
    kwargs...
) where {ModelType<:Sampleable, SamplerType<:AbstractSampler}
    # Do nothing.
    @warn "No sample_end! function has been implemented for objects
           of types $(typeof(ℓ)) and $(typeof(s))"
end

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

end # module Interface
