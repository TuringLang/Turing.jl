# TODO: Implement additional checks for certain samplers, e.g.
# HMC not supporting discrete parameters.
function _check_model(model::DynamicPPL.Model)
    new_context = DynamicPPL.setleafcontext(model.context, DynamicPPL.InitContext())
    new_model = DynamicPPL.contextualize(model, new_context)
    return DynamicPPL.check_model(new_model, VarInfo(); error_on_failure=true)
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
