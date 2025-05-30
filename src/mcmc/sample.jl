module Sample

using AbstractMCMC: AbstractMCMC, AbstractMCMCEnsemble, AbstractModel
using DynamicPPL: DynamicPPL, Sampler, Model
using MCMCChains: MCMCChains
using Random: Random, AbstractRNG
using ..Inference: Hamiltonian, InferenceAlgorithm, RepeatSampler
using ...Turing: PROGRESS

# TODO: Implement additional checks for certain samplers, e.g.
# HMC not supporting discrete parameters.
function _check_model(model::DynamicPPL.Model)
    return DynamicPPL.check_model(model; error_on_failure=true)
end
function _check_model(model::DynamicPPL.Model, alg::InferenceAlgorithm)
    return _check_model(model)
end

#########################################
# Default definitions for the interface #
#########################################

function AbstractMCMC.sample(
    model::AbstractModel, alg::InferenceAlgorithm, N::Integer; kwargs...
)
    return AbstractMCMC.sample(Random.default_rng(), model, alg, N; kwargs...)
end

function AbstractMCMC.sample(
    rng::AbstractRNG,
    model::AbstractModel,
    alg::InferenceAlgorithm,
    N::Integer;
    check_model::Bool=true,
    kwargs...,
)
    check_model && _check_model(model, alg)
    return AbstractMCMC.sample(rng, model, Sampler(alg), N; kwargs...)
end

function AbstractMCMC.sample(
    model::AbstractModel,
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
    rng::AbstractRNG,
    model::AbstractModel,
    alg::InferenceAlgorithm,
    ensemble::AbstractMCMC.AbstractMCMCEnsemble,
    N::Integer,
    n_chains::Integer;
    check_model::Bool=true,
    kwargs...,
)
    check_model && _check_model(model, alg)
    return AbstractMCMC.sample(rng, model, Sampler(alg), ensemble, N, n_chains; kwargs...)
end

function AbstractMCMC.sample(
    rng::AbstractRNG,
    model::AbstractModel,
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

end # module
