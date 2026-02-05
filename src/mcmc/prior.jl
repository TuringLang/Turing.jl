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
    discard_sample=false,
    kwargs...,
)
    accs = DynamicPPL.AccumulatorTuple((
        DynamicPPL.ValuesAsInModelAccumulator(true),
        DynamicPPL.LogPriorAccumulator(),
        DynamicPPL.LogLikelihoodAccumulator(),
    ))
    vi = DynamicPPL.OnlyAccsVarInfo(accs)
    _, vi = DynamicPPL.init!!(
        rng, model, vi, DynamicPPL.InitFromPrior(), DynamicPPL.UnlinkAll()
    )
    transition = discard_sample ? nothing : DynamicPPL.ParamsWithStats(vi)
    return transition, nothing
end
