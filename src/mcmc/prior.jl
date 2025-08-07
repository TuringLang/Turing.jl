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
    # TODO(DPPL0.38/penelopeysm): replace this entire thing with init!!
    #
    # `vi` is a VarInfo from the previous step so already has all the
    # right accumulators and stuff. The only thing we need to change is to make
    # sure that the old values are overwritten when we resample.
    #
    # Note also that the values in the Transition (and hence the chain) are not
    # obtained from the VarInfo's metadata itself, but are instead obtained
    # from the ValuesAsInModelAccumulator, which is cleared in the evaluate!!
    # call. Thus, the actual values in the VarInfo's metadata don't matter:
    # we only set the del flag here to make sure that new values are sampled
    # (and thus new values enter VAIMAcc), rather than the old ones being
    # reused during the evaluation. Yes, SampleFromPrior really sucks.
    for vn in keys(vi)
        DynamicPPL.set_flag!(vi, vn, "del")
    end
    sampling_model = DynamicPPL.contextualize(
        model, DynamicPPL.SamplingContext(rng, DynamicPPL.SampleFromPrior(), model.context)
    )
    _, vi = DynamicPPL.evaluate!!(sampling_model, vi)
    return Transition(model, vi, nothing; reevaluate=false), vi
end
