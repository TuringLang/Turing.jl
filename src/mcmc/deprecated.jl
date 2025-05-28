module SampleDeprecated

using AbstractMCMC: AbstractMCMC, AbstractMCMCEnsemble
using DynamicPPL: DynamicPPL, Sampler, Model
using MCMCChains: MCMCChains
using Random: Random
using ..Inference: Hamiltonian, InferenceAlgorithm, RepeatSampler
using ...Turing: PROGRESS

##########################################################
# OLD DEFINITIONS - Need to keep these for compatibility #
# so that unfixed samplers still work.                   #
# TODO: Remove these when all samplers are updated.      #
##########################################################

function AbstractMCMC.sample(model::Model, alg::InferenceAlgorithm, N::Integer; kwargs...)
    return AbstractMCMC.sample(Random.default_rng(), model, alg, N; kwargs...)
end

function AbstractMCMC.sample(
    rng::Random.AbstractRNG,
    model::Model,
    alg::InferenceAlgorithm,
    N::Integer;
    check_model::Bool=true,
    kwargs...,
)
    check_model && DynamicPPL.check_model(model)
    return AbstractMCMC.sample(rng, model, Sampler(alg), N; kwargs...)
end

function AbstractMCMC.sample(
    model::Model,
    alg::InferenceAlgorithm,
    ensemble::AbstractMCMC.AbstractMCMCEnsemble,
    N::Integer,
    n_chains::Integer;
    kwargs...,
)
    return AbstractMCMC.sample(
        Random.default_rng(), model, alg, ensemble, N, n_chains; kwargs...
    )
end

function AbstractMCMC.sample(
    rng::Random.AbstractRNG,
    model::Model,
    alg::InferenceAlgorithm,
    ensemble::AbstractMCMC.AbstractMCMCEnsemble,
    N::Integer,
    n_chains::Integer;
    check_model::Bool=true,
    kwargs...,
)
    check_model && check_model(model)
    return AbstractMCMC.sample(rng, model, Sampler(alg), ensemble, N, n_chains; kwargs...)
end

function AbstractMCMC.sample(
    rng::Random.AbstractRNG,
    model::Model,
    sampler::Union{Sampler{<:InferenceAlgorithm},RepeatSampler},
    ensemble::AbstractMCMC.AbstractMCMCEnsemble,
    N::Integer,
    n_chains::Integer;
    chain_type=MCMCChains.Chains,
    progress=PROGRESS[],
    kwargs...,
)
    return AbstractMCMC.mcmcsample(
        rng,
        model,
        sampler,
        ensemble,
        N,
        n_chains;
        chain_type=chain_type,
        progress=progress,
        kwargs...,
    )
end

##########################################################
# initialstep for fixed samplers                         #
# We need to keep these around because the Gibbs sampler #
# still calls them.                                      #
# TODO: Remove these when all samplers are updated.      #
##########################################################

function DynamicPPL.initialstep(
    rng::Random.AbstractRNG,
    model::Model,
    spl::Sampler{<:Hamiltonian},
    vi_original::DynamicPPL.AbstractVarInfo;
    kwargs...,
)
    vi = DynamicPPL.link(vi_original, model)
    ldf = DynamicPPL.LogDensityFunction(
        model,
        vi,
        DynamicPPL.SamplingContext(rng, spl, DynamicPPL.leafcontext(model.context));
        adtype=spl.alg.adtype,
    )
    return AbstractMCMC.step(rng, ldf, spl; kwargs...)
end

end # module
