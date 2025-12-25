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
    vi = DynamicPPL.OnlyAccsVarInfo(accs)
    _, vi = DynamicPPL.init!!(rng, model, vi, DynamicPPL.InitFromPrior())
    return DynamicPPL.ParamsWithStats(vi), nothing
end
