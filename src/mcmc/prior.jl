"""
    Prior()

Algorithm for sampling from the prior.
"""
struct Prior <: AbstractSampler end

function AbstractMCMC.step(
    rng::Random.AbstractRNG, model::DynamicPPL.Model, sampler::Prior; kwargs...
)
    accs = DynamicPPL.AccumulatorTuple((
        DynamicPPL.ValuesAsInModelAccumulator(true),
        DynamicPPL.LogPriorAccumulator(),
        DynamicPPL.LogLikelihoodAccumulator(),
    ))
    sampling_model = DynamicPPL.setleafcontext(
        model, DynamicPPL.InitContext(rng, DynamicPPL.InitFromPrior())
    )
    vi = DynamicPPL.OnlyAccsVarInfo(accs)
    _, vi = DynamicPPL.evaluate!!(sampling_model, vi)
    return Transition(sampling_model, vi, nothing; reevaluate=false), (sampling_model, vi)
end

function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::DynamicPPL.Model,
    sampler::Prior,
    state::Tuple{DynamicPPL.Model,DynamicPPL.Experimental.OnlyAccsVarInfo};
    kwargs...,
)
    model, vi = state
    _, vi = DynamicPPL.evaluate!!(model, vi)
    return Transition(model, vi, nothing; reevaluate=false), (model, vi)
end
