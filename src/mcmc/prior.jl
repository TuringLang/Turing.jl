"""
    Prior()

Algorithm for sampling from the prior.
"""
struct Prior <: AbstractSampler end

function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::DynamicPPL.Model,
    sampler::Prior,
    state=nothing;
    kwargs...,
)
    accs = DynamicPPL.AccumulatorTuple((
        DynamicPPL.ValuesAsInModelAccumulator(true),
        DynamicPPL.LogPriorAccumulator(),
        DynamicPPL.LogLikelihoodAccumulator(),
    ))
    _, vi = DynamicPPL.fast_evaluate!!(rng, model, InitFromPrior(), accs)
    return DynamicPPL.ParamsWithStats(vi), nothing
end
