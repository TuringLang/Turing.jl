module Sample

using AbstractMCMC: AbstractMCMC, AbstractMCMCEnsemble
using DynamicPPL: DynamicPPL, Sampler, LogDensityFunction, Model
using MCMCChains: Chains
using Random: Random
using ..Inference:
    InferenceAlgorithm,
    RepeatSampler,
    update_sample_kwargs,
    get_adtype,
    requires_unconstrained_space
using ...Turing: PROGRESS

# This file contains the basic methods for `AbstractMCMC.sample`.
#
# The overall aim is that users can call
#
#    sample(::Model, ::InferenceAlgorithm, N)
#
# and have it be (eventually) forwarded to
#
#    sample(::LogDensityFunction, ::Sampler{InferenceAlgorithm}, N) 
#
# The former method is more convenient for most users, and has been the 'default'
# API in Turing. The latter method is what really needs to be used under the hood,
# because a Model on its own does not fully specify how the log-density should be
# evaluated (only a LogDensityFunction has that information).
#
# Thus, advanced users who want to customise the way their model is executed (e.g.
# by using different types of VarInfo) can construct their own LogDensityFunction
# and call `sample(ldf, spl, N)` themselves.

# The main method: without ensemble sampling
# NOTE: When updating this method, please make sure to also update the
# corresponding one with ensemble sampling, right below it.
function AbstractMCMC.sample(
    rng::Random.AbstractRNG,
    ldf::LogDensityFunction,
    spl::Union{Sampler{<:InferenceAlgorithm},RepeatSampler},
    N::Integer;
    check_model::Bool=true,
    chain_type=Chains,
    progress=PROGRESS[],
    resume_from=nothing,
    initial_state=DynamicPPL.loadstate(resume_from),
    kwargs...,
)
    # TODO: Right now, only generic checks are run. We could in principle
    # specialise this to check for e.g. discrete variables with HMC
    check_model && DynamicPPL.check_model(ldf.model; error_on_failure=true)
    # Some samplers need to update the kwargs with additional information,
    # e.g. HMC.
    new_kwargs = update_sample_kwargs(spl, N, kwargs)
    # Forward to the main sampling function
    return AbstractMCMC.mcmcsample(
        rng,
        ldf,
        spl,
        N;
        initial_state=initial_state,
        chain_type=chain_type,
        progress=progress,
        new_kwargs...,
    )
end

# The main method: with ensemble sampling
# NOTE: When updating this method, please make sure to also update the
# corresponding one without ensemble sampling, right above it.
function AbstractMCMC.sample(
    rng::Random.AbstractRNG,
    ldf::LogDensityFunction,
    spl::Union{Sampler{<:InferenceAlgorithm},RepeatSampler},
    ensemble::AbstractMCMC.AbstractMCMCEnsemble,
    N::Integer,
    n_chains::Integer;
    check_model::Bool=true,
    chain_type=Chains,
    progress=PROGRESS[],
    resume_from=nothing,
    initial_state=DynamicPPL.loadstate(resume_from),
    kwargs...,
)
    # TODO: Right now, only generic checks are run. We could in principle
    # specialise this to check for e.g. discrete variables with HMC
    check_model && DynamicPPL.check_model(ldf.model; error_on_failure=true)
    # Some samplers need to update the kwargs with additional information,
    # e.g. HMC.
    new_kwargs = update_sample_kwargs(spl, N, kwargs)
    # Forward to the main sampling function
    return AbstractMCMC.mcmcsample(
        rng,
        ldf,
        spl,
        ensemble,
        N,
        n_chains;
        initial_state=initial_state,
        chain_type=chain_type,
        progress=progress,
        kwargs...,
    )
end

####################################################
### The rest of this file is boring boilerplate. ###
####################################################

function AbstractMCMC.sample(
    model_or_ldf::Union{Model,LogDensityFunction},
    alg_or_spl::Union{InferenceAlgorithm,Sampler{<:InferenceAlgorithm}},
    N::Integer;
    kwargs...,
)
    return AbstractMCMC.sample(Random.default_rng(), model_or_ldf, alg_or_spl, N; kwargs...)
end

function AbstractMCMC.sample(
    rng::Random.AbstractRNG,
    ldf::LogDensityFunction,
    alg::InferenceAlgorithm,
    N::Integer;
    kwargs...,
)
    return AbstractMCMC.sample(rng, ldf, Sampler(alg), N; kwargs...)
end

function AbstractMCMC.sample(
    rng::Random.AbstractRNG, model::Model, alg::InferenceAlgorithm, N::Integer; kwargs...
)
    return AbstractMCMC.sample(rng, model, Sampler(alg), N; kwargs...)
end

function AbstractMCMC.sample(
    rng::Random.AbstractRNG,
    model::Model,
    spl::Sampler{<:InferenceAlgorithm},
    N::Integer;
    kwargs...,
)
    initial_params = get(kwargs, :initial_params, nothing)
    link = requires_unconstrained_space(spl)
    vi = DynamicPPL.default_varinfo(rng, model, spl, initial_params, link)
    ldf = LogDensityFunction(model; adtype=get_adtype(spl))
    return AbstractMCMC.sample(rng, ldf, spl, N; kwargs...)
end

function AbstractMCMC.sample(
    model_or_ldf::Union{Model,LogDensityFunction},
    alg_or_spl::Union{InferenceAlgorithm,Sampler{<:InferenceAlgorithm}},
    ensemble::AbstractMCMC.AbstractMCMCEnsemble,
    N::Integer,
    n_chains::Integer;
    kwargs...,
)
    return AbstractMCMC.sample(
        Random.default_rng(), model_or_ldf, alg_or_spl, ensemble, N, n_chains; kwargs...
    )
end

function AbstractMCMC.sample(
    rng::Random.AbstractRNG,
    ldf::LogDensityFunction,
    alg::InferenceAlgorithm,
    ensemble::AbstractMCMC.AbstractMCMCEnsemble,
    N::Integer,
    n_chains::Integer;
    kwargs...,
)
    return AbstractMCMC.sample(rng, ldf, Sampler(alg), ensemble, N, n_chains; kwargs...)
end

function AbstractMCMC.sample(
    rng::Random.AbstractRNG,
    model::Model,
    alg::InferenceAlgorithm,
    ensemble::AbstractMCMC.AbstractMCMCEnsemble,
    N::Integer,
    n_chains::Integer;
    kwargs...,
)
    return AbstractMCMC.sample(rng, model, Sampler(alg), ensemble, N, n_chains; kwargs...)
end

function AbstractMCMC.sample(
    rng::Random.AbstractRNG,
    model::Model,
    spl::Sampler{<:InferenceAlgorithm},
    ensemble::AbstractMCMC.AbstractMCMCEnsemble,
    N::Integer,
    n_chains::Integer;
    kwargs...,
)
    initial_params = get(kwargs, :initial_params, nothing)
    link = requires_unconstrained_space(spl)
    vi = DynamicPPL.default_varinfo(rng, model, spl, initial_params, link)
    ldf = LogDensityFunction(model; adtype=get_adtype(spl))
    return AbstractMCMC.sample(rng, ldf, spl, ensemble, N, n_chains; kwargs...)
end

end # module
