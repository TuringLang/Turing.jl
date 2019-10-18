module Interface

import Distributions: sample, Sampleable
import Random: GLOBAL_RNG, AbstractRNG, seed!
import MCMCChains: Chains, chainscat
import ProgressMeter

export AbstractSampler,
       AbstractTransition,
       AbstractCallback,
       init_callback,
       callback,
       transitions_init,
       transition_type,
       sample_init!,
       sample_end!,
       sample,
       psample,
       Sampleable,
       AbstractRNG,
       Chains,
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
    AbstractCallback

An `AbstractCallback` types is a supertype to be inherited from if you want to use custom callback 
functionality. This is used to report sampling progress such as parameters calculated, remaining
samples to run, or even plot graphs if you so choose.

In order to implement callback functionality, you need the following:

- A mutable struct that is a subtype of `AbstractCallback`
- An overload of the `init_callback` function
- An overload of the `callback` function
"""
abstract type AbstractCallback end

"""
    NoCallback()

This disables the callback functionality in the event that you wish to 
implement your own callback or reporting.
"""
mutable struct NoCallback <: AbstractCallback end

"""
    DefaultCallback(N::Int)

The default callback struct which uses `ProgressMeter`.
"""
mutable struct DefaultCallback{
    ProgType<:ProgressMeter.AbstractProgress
} <: AbstractCallback
    p :: ProgType
end

DefaultCallback(N::Int) = DefaultCallback(ProgressMeter.Progress(N, 1))

function init_callback(
    rng::AbstractRNG,
    ℓ::ModelType,
    s::SamplerType,
    N::Integer;
    kwargs...
) where {ModelType<:Sampleable, SamplerType<:AbstractSampler}
    return DefaultCallback(N)
end

"""
    _generate_callback(
        rng::AbstractRNG,
        ℓ::ModelType,
        s::SamplerType,
        N::Integer;
        progress_style=:default,
        kwargs...
    )

`_generate_callback` uses a `progress_style` keyword argument to determine
which progress meter style should be used. This function is strictly internal
and is not meant to be overloaded. If you intend to add a custom `AbstractCallback`,
you should overload `init_callback` instead.

Options for `progress_style` include:

    - `:default` which returns the result of `init_callback`
    - `false` or `:disable` which returns a `NoCallback`
    - `:plain` which returns the default, simple `DefaultCallback`.
"""
function _generate_callback(
    rng::AbstractRNG,
    ℓ::ModelType,
    s::SamplerType,
    N::Integer;
    progress_style=:default,
    kwargs...
) where {ModelType<:Sampleable, SamplerType<:AbstractSampler}
    if progress_style == :default
        return init_callback(rng, ℓ, s, N; kwargs...)
    elseif progress_style == false || progress_style == :disable
        return NoCallback()
    elseif progress_style == :plain
        return DefaultCallback(N)
    else
        throw(ArgumentError("Keyword argument $progress_style is not recognized."))
    end
end

"""
    sample(
        rng::AbstractRNG,
        ℓ::Sampleable,
        s::AbstractSampler,
        N::Integer;
        kwargs...
    )

    sample(
        ℓ::ModelType,
        s::SamplerType,
        N::Integer;
        kwargs...
    )

`sample` returns an `MCMCChains.Chains` object containing `N` samples from a given model and
sampler. You may pass in any additional arguments through the use of keyword arguments.
"""
function sample(
    ℓ::ModelType,
    s::SamplerType,
    N::Integer;
    kwargs...
) where {ModelType<:Sampleable, SamplerType<:AbstractSampler}
    return sample(GLOBAL_RNG, ℓ, s, N; kwargs...)
end

function sample(
    rng::AbstractRNG,
    ℓ::ModelType,
    s::SamplerType,
    N::Integer;
    progress::Bool=true,
    kwargs...
) where {ModelType<:Sampleable, SamplerType<:AbstractSampler}
    # Perform any necessary setup.
    sample_init!(rng, ℓ, s, N; kwargs...)

    # Preallocate the TransitionType vector.
    ts = transitions_init(rng, ℓ, s, N; kwargs...)

    # Add a progress meter.
    cb = progress ? _generate_callback(rng, ℓ, s, N; kwargs...) : nothing

    # Step through the sampler.
    for i=1:N
        if i == 1
            ts[i] = step!(rng, ℓ, s, N; iteration=i, kwargs...)
        else
            ts[i] = step!(rng, ℓ, s, N, ts[i-1]; iteration=i, kwargs...)
        end

        # Run a callback function.
        progress && callback(rng, ℓ, s, N, i, ts[i], cb; kwargs...)
    end

    # Wrap up the sampler, if necessary.
    sample_end!(rng, ℓ, s, N, ts; kwargs...)

    return Chains(rng, ℓ, s, N, ts; kwargs...)
end

"""
    sample_init!(
        rng::AbstractRNG,
        ℓ::ModelType,
        s::SamplerType,
        N::Integer;
        kwargs...
    )

Performs whatever initial setup is required for your sampler. This function is not intended
to return any value -- any set up should mutate the sampler or the model type in-place.

A common use for `sample_init!` might be to instantiate a particle field for later use,
or find an initial step size for a Hamiltonian sampler.
"""
function sample_init!(
    rng::AbstractRNG,
    ℓ::ModelType,
    s::SamplerType,
    N::Integer;
    debug::Bool=false,
    kwargs...
) where {ModelType<:Sampleable, SamplerType<:AbstractSampler}
    # Do nothing.
    debug && @warn "No sample_init! function has been implemented for objects
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

Performs whatever finalizing the sampler requires. This function is not intended
to return any value -- any set up should mutate the sampler or the model type in-place.

`sample_end!` is useful in cases where you might like to perform some transformation 
on your vector of `AbstractTransitions`, save your sampler struct to disk, or otherwise
perform any clean-up or finalization.
"""
function sample_end!(
    rng::AbstractRNG,
    ℓ::ModelType,
    s::SamplerType,
    N::Integer,
    ts::Vector{TransitionType};
    debug::Bool=false,
    kwargs...
) where {
    ModelType<:Sampleable,
    SamplerType<:AbstractSampler,
    TransitionType<:AbstractTransition
}
    # Do nothing.
    debug && @warn "No sample_end! function has been implemented for objects
           of types $(typeof(ℓ)) and $(typeof(s))"
end

"""
    step!(
        rng::AbstractRNG,
        ℓ::Sampleable,
        s::AbstractSampler,
        N::Integer;
        kwargs...
    )

    step!(
        rng::AbstractRNG,
        ℓ::Sampleable,
        s::AbstractSampler;
        kwargs...
    )

    step!(
        rng::AbstractRNG,
        ℓ::Sampleable,
        s::AbstractSampler,
        N::Integer,
        t::AbstractTransition;
        kwargs...
    )

Returns a single `AbstractTransition` drawn using the provided random number generator, 
model, and sampler. `step!` is the function that performs inference, and it is how
a model moves from one sample to another.

`step!` may modify the model or the sampler in-place. As an example, you may have a state
variable in your sampler that contains a vector of particles or some other value that
does not need to be included in the `AbstractTransition` struct returned.

Every `step!` call after the first has access to the previous `AbstractTransition`.
"""
function step!(
    rng::AbstractRNG,
    ℓ::ModelType,
    s::SamplerType,
    N::Integer;
    debug::Bool=false,
    kwargs...
) where {ModelType<:Sampleable, SamplerType<:AbstractSampler}
    # Do nothing.
    debug && @warn "No step! function has been implemented for objects of types \n- $(typeof(ℓ)) \n- $(typeof(s))"
end

function step!(
    rng::AbstractRNG,
    ℓ::ModelType,
    s::SamplerType;
    kwargs...
) where {ModelType<:Sampleable, SamplerType<:AbstractSampler}
    return step!(rng, ℓ, s, 1; kwargs...)
end

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
    # @warn "No step! function has been implemented for objects
    #        of types $(typeof(ℓ)) and $(typeof(s))"
    return step!(rng, ℓ, s, N; kwargs...)
end

function step!(
    rng::AbstractRNG,
    ℓ::ModelType,
    s::SamplerType,
    N::Integer,
    t::Nothing;
    debug::Bool=true,
    kwargs...
) where {ModelType<:Sampleable,
    SamplerType<:AbstractSampler,
    TransitionType<:AbstractTransition
}
    debug && @warn "No transition type passed in, running normal step! function."
    return step!(rng, ℓ, s, N; kwargs...)
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
    return Vector{transition_type(s)}(undef, N)
end

"""
    callback(
        rng::AbstractRNG,
        ℓ::ModelType,
        s::SamplerType,
        N::Integer,
        iteration::Integer,
        cb::CallbackType;
        kwargs...
    )

`callback` is called after every sample run, and allows you to run some function on a 
subtype of `AbstractCallback`. Typically this is used to increment a progress meter, show a 
plot of parameter draws, or otherwise provide information about the sampling process to the user.

By default, `ProgressMeter` is used to show the number of samples remaning.
"""
function callback(
    rng::AbstractRNG,
    ℓ::ModelType,
    s::SamplerType,
    N::Integer,
    iteration::Integer,
    t::TransitionType,
    cb::CallbackType;
    kwargs...
) where {
    ModelType<:Sampleable,
    SamplerType<:AbstractSampler,
    CallbackType<:AbstractCallback,
    TransitionType<:AbstractTransition
}
    # Default callback behavior.
    ProgressMeter.next!(cb.p)
end

function callback(
    rng::AbstractRNG,
    ℓ::ModelType,
    s::SamplerType,
    N::Integer,
    iteration::Integer,
    t::TransitionType,
    cb::NoCallback;
    kwargs...
) where {
    ModelType<:Sampleable,
    SamplerType<:AbstractSampler,
    TransitionType<:AbstractTransition
}
    # Do nothing.
end

"""
    transition_type(s::AbstractSampler)

Return the type of `AbstractTransition` that is to be returned by an 
`AbstractSampler` after each `step!` call. 
"""
transition_type(s::AbstractSampler) = AbstractTransition

"""
    psample(ℓ::ModelType, s::SamplerType, N::Integer, n_chains::Integer; kwargs...)
    psample(rng::AbstractRNG, ℓ::ModelType, s::SamplerType, N::Integer, n_chains::Integer; kwargs...)

Samples `n_chains` using the threads available, and combines the returned chains into a single 
`Chains` struct.

By default, the model and sampler types are deep copied for each thread to prevent contamination
between threads. 
"""
function psample(
    ℓ::ModelType,
    s::SamplerType,
    N::Integer,
    n_chains::Integer;
    kwargs...
) where {
    ModelType <: Sampleable,
    SamplerType <: AbstractSampler
}
    return psample(GLOBAL_RNG, ℓ, s, N, n_chains; kwargs...)
end

function psample(
    rng::AbstractRNG,
    ℓ::ModelType,
    s::SamplerType,
    N::Integer,
    n_chains::Integer;
    kwargs...
) where {
    ModelType <: Sampleable,
    SamplerType <: AbstractSampler
}
    # Set up a chains vector.
    chains = []

    # Create a thread lock. Used to add append chains to the vector.
    c_lock = Threads.SpinLock()

    # Create a seed for each chain using rng.
    seeds = rand(rng, UInt, n_chains)

    Threads.@threads for i in 1:n_chains
        # Generate a new RNG using the pre-made seeds.
        sub_rng = seed!(seeds[i])

        # Sample a chain.
        chain = sample(sub_rng, deepcopy(ℓ), deepcopy(s), N; progress=false, kwargs...)
        
        # Acquire the lock, add the chain to the vector.
        lock(c_lock) do
            push!(chains, (order=i, chain=chain))
        end
    end

    # Sort the chains by seed order, to ensure that all chains are always in the same order.
    sort!(chains, by=x -> x.order)

    # Concatenate the chains together.
    return chainscat([c.chain for c in chains]...)
end

end # module Interface
