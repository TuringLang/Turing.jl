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

function RepeatSampler(alg::InferenceAlgorithm, num_repeat::Int)
    return RepeatSampler(Sampler(alg), num_repeat)
end

drop_space(rs::RepeatSampler) = RepeatSampler(drop_space(rs.sampler), rs.num_repeat)
getADType(spl::RepeatSampler) = getADType(spl.sampler)
DynamicPPL.default_chain_type(sampler::RepeatSampler) = default_chain_type(sampler.sampler)
DynamicPPL.getspace(spl::RepeatSampler) = getspace(spl.sampler)
DynamicPPL.inspace(vn::VarName, spl::RepeatSampler) = inspace(vn, spl.sampler)

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
