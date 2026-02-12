"""
    RepeatSampler <: AbstractMCMC.AbstractSampler

A `RepeatSampler` is a container for a sampler and a number of times to repeat it.

# Fields
$(FIELDS)

# Examples
```julia
repeated_sampler = RepeatSampler(sampler, 10)
AbstractMCMC.step(rng, model, repeated_sampler) # take 10 steps of `sampler`
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

function setparams_varinfo!!(
    model::DynamicPPL.Model,
    sampler::RepeatSampler,
    state,
    params::DynamicPPL.AbstractVarInfo,
)
    return setparams_varinfo!!(model, sampler.sampler, state, params)
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

function Turing.Inference.init_strategy(spl::RepeatSampler)
    return Turing.Inference.init_strategy(spl.sampler)
end

function AbstractMCMC.sample(
    rng::AbstractRNG,
    model::DynamicPPL.Model,
    sampler::RepeatSampler,
    N::Integer;
    check_model=true,
    initial_params=Turing.Inference.init_strategy(sampler),
    chain_type=DEFAULT_CHAIN_TYPE,
    progress=PROGRESS[],
    kwargs...,
)
    check_model && Turing._check_model(model, sampler)
    return AbstractMCMC.mcmcsample(
        rng,
        model,
        sampler,
        N;
        initial_params=Turing._convert_initial_params(initial_params),
        chain_type=chain_type,
        progress=progress,
        kwargs...,
    )
end

function AbstractMCMC.sample(
    rng::AbstractRNG,
    model::DynamicPPL.Model,
    sampler::RepeatSampler,
    ensemble::AbstractMCMC.AbstractMCMCEnsemble,
    N::Integer,
    n_chains::Integer;
    check_model=true,
    initial_params=fill(Turing.Inference.init_strategy(sampler), n_chains),
    chain_type=DEFAULT_CHAIN_TYPE,
    progress=PROGRESS[],
    kwargs...,
)
    check_model && Turing._check_model(model, sampler)
    return AbstractMCMC.mcmcsample(
        rng,
        model,
        sampler,
        ensemble,
        N,
        n_chains;
        initial_params=map(Turing._convert_initial_params, initial_params),
        chain_type=chain_type,
        progress=progress,
        kwargs...,
    )
end
