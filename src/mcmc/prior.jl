"""
    Prior()

Algorithm for sampling from the prior.
"""
struct Prior <: InferenceAlgorithm end

function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::DynamicPPL.Model,
    sampler::DynamicPPL.Sampler{<:Prior};
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
    vi = DynamicPPL.typed_varinfo(vi)
    return Transition(model, vi, nothing; reevaluate=false), vi
end

function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::DynamicPPL.Model,
    sampler::DynamicPPL.Sampler{<:Prior},
    vi::AbstractVarInfo;
    kwargs...,
)
    # TODO(DPPL0.38/penelopeysm): replace the entire thing with init!!
    # `vi` is a VarInfo from the previous step so already has all the
    # right accumulators and stuff. The only thing we need to change is
    # to make sure that the old values are overwritten...
    for vn in keys(vi)
        DynamicPPL.set_flag!(vi, vn, "del")
    end
    sampling_model = DynamicPPL.contextualize(
        model, DynamicPPL.SamplingContext(rng, DynamicPPL.SampleFromPrior(), model.context)
    )
    _, vi = DynamicPPL.evaluate!!(sampling_model, vi)
    return Transition(model, vi, nothing; reevaluate=false), vi
end
