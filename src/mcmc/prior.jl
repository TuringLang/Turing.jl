"""
    Prior()

Algorithm for sampling from the prior.
"""
struct Prior <: InferenceAlgorithm end

function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::DynamicPPL.Model,
    sampler::DynamicPPL.Sampler{<:Prior},
    state=nothing;
    kwargs...,
)
    vi = last(
        DynamicPPL.evaluate!!(
            model,
            VarInfo(),
            SamplingContext(rng, DynamicPPL.SampleFromPrior(), DynamicPPL.PriorContext()),
        ),
    )
    return vi, nothing
end

DynamicPPL.default_chain_type(sampler::Prior) = MCMCChains.Chains
