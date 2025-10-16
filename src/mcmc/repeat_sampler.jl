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

function RepeatSampler(alg::AbstractSampler, num_repeat::Int)
    return RepeatSampler(alg, num_repeat)
end

function setparams_varinfo!!(model::DynamicPPL.Model, sampler::RepeatSampler, state, params)
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

function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::AbstractMCMC.AbstractModel,
    sampler::RepeatSampler,
    state;
    kwargs...,
)
    transition, state = AbstractMCMC.step(rng, model, sampler.sampler, state; kwargs...)
    for _ in 2:(sampler.num_repeat)
        transition, state = AbstractMCMC.step(rng, model, sampler.sampler, state; kwargs...)
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
    kwargs...,
)
    transition, state = AbstractMCMC.step_warmup(
        rng, model, sampler.sampler, state; kwargs...
    )
    for _ in 2:(sampler.num_repeat)
        transition, state = AbstractMCMC.step_warmup(
            rng, model, sampler.sampler, state; kwargs...
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
    initial_params=Turing.Inference.init_strategy(sampler),
    chain_type=DEFAULT_CHAIN_TYPE,
    progress=PROGRESS[],
    kwargs...,
)
    return AbstractMCMC.mcmcsample(
        rng,
        model,
        sampler,
        N;
        initial_params=_convert_initial_params(initial_params),
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
    initial_params=fill(Turing.Inference.init_strategy(sampler), n_chains),
    chain_type=DEFAULT_CHAIN_TYPE,
    progress=PROGRESS[],
    kwargs...,
)
    return AbstractMCMC.mcmcsample(
        rng,
        model,
        sampler,
        ensemble,
        N,
        n_chains;
        initial_params=map(_convert_initial_params, initial_params),
        chain_type=chain_type,
        progress=progress,
        kwargs...,
    )
end
