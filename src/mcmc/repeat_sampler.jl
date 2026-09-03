"""
    RepeatSampler <: AbstractMCMC.AbstractSampler

A `RepeatSampler` is a container for a sampler and a number of times to repeat it.

# Fields
$(FIELDS)

# Examples
```julia
repeated_sampler = RepeatSampler(sampler, 10)
# The initial step is a single step of `sampler`; it is the steps from a state that repeat.
_, state = AbstractMCMC.step(rng, model, repeated_sampler)
AbstractMCMC.step(rng, model, repeated_sampler, state) # take 10 steps of `sampler`
```
"""
struct RepeatSampler{S<:AbstractMCMC.AbstractSampler} <: AbstractMCMC.AbstractSampler
    "The sampler to repeat"
    sampler::S
    "The number of times to repeat the sampler"
    num_repeat::Int

    function RepeatSampler(sampler::S, num_repeat::Int) where {S}
        @assert num_repeat > 0
        return new{S}(sampler, num_repeat)
    end
end

function _variable_set_checked(sampler::RepeatSampler)
    return RepeatSampler(_variable_set_checked(sampler.sampler), sampler.num_repeat)
end

function gibbs_update_state!!(
    sampler::RepeatSampler,
    state,
    model::DynamicPPL.Model,
    global_vnt::DynamicPPL.VarNamedTuple,
)
    return gibbs_update_state!!(sampler.sampler, state, model, global_vnt)
end

function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::AbstractMCMC.AbstractModel,
    sampler::RepeatSampler;
    kwargs...,
)
    return AbstractMCMC.step(rng, model, sampler.sampler; kwargs...)
end
# The following method needed for method ambiguity resolution.
# TODO(penelopeysm): Remove this method once the default `AbstractMCMC.step(rng,
# ::DynamicPPL.Model, ::AbstractSampler)` method in `src/mcmc/abstractmcmc.jl` is removed.
function AbstractMCMC.step(
    rng::Random.AbstractRNG, model::DynamicPPL.Model, sampler::RepeatSampler; kwargs...
)
    return AbstractMCMC.step(rng, model, sampler.sampler; kwargs...)
end

function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::AbstractMCMC.AbstractModel,
    sampler::RepeatSampler,
    state;
    discard_sample=false,
    kwargs...,
)
    discard_first_sample = discard_sample || sampler.num_repeat > 1
    transition, state = AbstractMCMC.step(
        rng, model, sampler.sampler, state; kwargs..., discard_sample=discard_first_sample
    )
    for i in 2:(sampler.num_repeat)
        discard_ith_sample = discard_sample || i < sampler.num_repeat
        transition, state = AbstractMCMC.step(
            rng, model, sampler.sampler, state; kwargs..., discard_sample=discard_ith_sample
        )
    end
    return transition, state
end

function AbstractMCMC.step_warmup(
    rng::Random.AbstractRNG,
    model::AbstractMCMC.AbstractModel,
    sampler::RepeatSampler;
    kwargs...,
)
    return AbstractMCMC.step_warmup(rng, model, sampler.sampler; kwargs...)
end

function AbstractMCMC.step_warmup(
    rng::Random.AbstractRNG,
    model::AbstractMCMC.AbstractModel,
    sampler::RepeatSampler,
    state;
    discard_sample=false,
    kwargs...,
)
    discard_first_sample = discard_sample || sampler.num_repeat > 1
    transition, state = AbstractMCMC.step_warmup(
        rng, model, sampler.sampler, state; kwargs..., discard_sample=discard_first_sample
    )
    for i in 2:(sampler.num_repeat)
        discard_ith_sample = discard_sample || i < sampler.num_repeat
        transition, state = AbstractMCMC.step_warmup(
            rng, model, sampler.sampler, state; kwargs..., discard_sample=discard_ith_sample
        )
    end
    return transition, state
end

# Need some extra leg work to make RepeatSampler work seamlessly with DynamicPPL models +
# samplers, instead of generic AbstractMCMC samplers.

function post_sample_hook(chain, spl::RepeatSampler; kwargs...)
    return post_sample_hook(chain, spl.sampler; kwargs...)
end

function allow_discrete_variables(spl::RepeatSampler)
    return allow_discrete_variables(spl.sampler)
end

function Turing.Inference.init_strategy(spl::RepeatSampler)
    return Turing.Inference.init_strategy(spl.sampler)
end

# No `AbstractMCMC.sample` methods here. `RepeatSampler` forwards `init_strategy` and
# `allow_discrete_variables`, so the generic methods in `abstractmcmc.jl` already do everything
# a specialisation could: they convert `initial_params`, check the model, thread `chain_type`
# and `verbose`, and call `post_sample_hook`. Copies of them drifted -- the ensemble one lost
# the hook, so a threaded run suppressed the wrapped sampler's divergence warning -- and the
# generic ensemble method additionally validates that `initial_params` has one element per
# chain, which the copy did not.
