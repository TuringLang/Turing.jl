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
    # TODO(DPPL0.37/penelopeysm): replace with init!! instead
    vi = last(
        DynamicPPL.evaluate!!(
            model, VarInfo(), DynamicPPL.SamplingContext(rng, DynamicPPL.SampleFromPrior())
        ),
    )
    # Need to manually construct the Transition here because we only
    # want to use the prior probability.
    xs = Turing.Inference.getparams(model, vi)
    lp = DynamicPPL.getlogprior(vi)
    return Transition(xs, lp, nothing)
end

DynamicPPL.default_chain_type(sampler::Prior) = MCMCChains.Chains
