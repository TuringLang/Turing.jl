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
    # TODO(DPPL0.38/penelopeysm): replace with init!!
    sampling_model = DynamicPPL.contextualize(
        model, DynamicPPL.SamplingContext(rng, DynamicPPL.SampleFromPrior(), model.context)
    )
    vi = VarInfo()
    vi = DynamicPPL.setaccs!!(
        vi,
        (
            DynamicPPL.ValuesAsInModelAccumulator(true),
            DynamicPPL.LogPriorAccumulator(),
            DynamicPPL.LogLikelihoodAccumulator(),
        ),
    )
    _, vi = DynamicPPL.evaluate!!(sampling_model, vi)
    return Transition(model, vi, nothing; reevaluate=false), nothing
end
