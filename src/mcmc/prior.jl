"""
    Prior()

Algorithm for sampling from the prior.

Every draw is an independent draw from the prior, so there is no starting point for
`initial_params` to set; passing one warns and has no effect. To hold a variable at a value,
`condition` or `fix` the model instead.
"""
struct Prior <: AbstractSampler end

function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::DynamicPPL.Model,
    sampler::Prior;
    initial_params=DynamicPPL.InitFromPrior(),
    discard_sample=false,
    kwargs...,
)
    # Every draw comes from the prior independently, so there is no starting point for one to
    # set: honouring `initial_params` would put a value that is not a prior draw into a sample
    # advertised as one. It is still ignored; only the silence has gone. Warned here, in the
    # method that starts a run, rather than in the one below that continues it, so that it is
    # said once per `sample` call -- `maxlog` would instead say it once per session, leaving
    # every later call as quiet as before.
    warn_initial_params_ignored(
        "`Prior()`", "draws every sample from the prior", initial_params
    )
    return _prior_step(rng, model, discard_sample)
end

function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::DynamicPPL.Model,
    sampler::Prior,
    state;
    discard_sample=false,
    kwargs...,
)
    return _prior_step(rng, model, discard_sample)
end

function _prior_step(rng::Random.AbstractRNG, model::DynamicPPL.Model, discard_sample::Bool)
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
