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
    initial_params=DynamicPPL.InitFromPrior(),
    discard_sample=false,
    kwargs...,
)
    # Every draw comes from the prior independently, so there is no starting point for one to
    # set. Honouring it would make the sampler's name untrue -- `InitFromUniform` would draw
    # from the init distribution, not the prior -- so it is refused in words rather than used.
    if !(initial_params isa DynamicPPL.InitFromPrior)
        @warn "`Prior()` draws every sample from the prior, so `initial_params` has no " *
            "effect and is being ignored." maxlog = 1
    end
    accs = DynamicPPL.AccumulatorTuple((
        DynamicPPL.RawValueAccumulator(true),
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
