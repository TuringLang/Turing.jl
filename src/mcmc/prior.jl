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
    vi = DynamicPPL.setaccs!!(
        DynamicPPL.VarInfo(),
        (
            DynamicPPL.ValuesAsInModelAccumulator(true),
            DynamicPPL.LogPriorAccumulator(),
            DynamicPPL.LogLikelihoodAccumulator(),
        ),
    )
    _, vi = DynamicPPL.init!!(model, vi, DynamicPPL.InitFromPrior())
    return Transition(model, vi, nothing; reevaluate=false), nothing
end
