module Sample

using AbstractMCMC: AbstractMCMC, AbstractMCMCEnsemble, AbstractSampler
using DynamicPPL: DynamicPPL, Sampler, LogDensityFunction, Model
using MCMCChains: Chains
using Random: Random
using ..Inference:
    Hamiltonian,
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

# Because this is a pain to implement all at once, we do it for one sampler at a time.
# This type tells us which samplers have been 'updated' to the new interface.

# TODO: Eventually, we want to broaden this to InferenceAlgorithm
const LDFCompatibleAlgorithm = Union{Hamiltonian}
# TODO: Eventually, we want to broaden this to
# Union{Sampler{<:InferenceAlgorithm},RepeatSampler}.
const LDFCompatibleSampler = Union{Sampler{<:LDFCompatibleAlgorithm}}

# The main method: without ensemble sampling
# NOTE: When updating this method, please make sure to also update the
# corresponding one with ensemble sampling, right below it.
function AbstractMCMC.sample(
    rng::Random.AbstractRNG,
    ldf::LogDensityFunction,
    spl::LDFCompatibleSampler,
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
    spl::LDFCompatibleSampler,
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

# This method should be in DynamicPPL. We will move it there when all the
# Turing samplers have been updated.
"""
    initialise_varinfo(rng, model, sampler[, context])

Return an initial varinfo object for the given `model` and `sampler`. If given,
the initial parameter values will be set in the varinfo object. Also performs
linking if requested.

# Arguments
- `rng::Random.AbstractRNG`: Random number generator.
- `model::Model`: Model for which we want to create a varinfo object.
- `sampler::AbstractSampler`: Sampler which will make use of the varinfo object.
- `initial_params::Union{AbstractVector,Nothing}`: Initial parameter values to
be set in the varinfo object. Note that these should be given in unconstrained
space.
- `link::Bool`: Whether to link the varinfo.
- `context::AbstractContext`: Context in which the model is evaluated. Defaults
to `DefaultContext()`.

# Returns
- `AbstractVarInfo`: Default varinfo object for the given `model` and `sampler`.
"""
function initialise_varinfo(
    rng::Random.AbstractRNG,
    model::Model,
    sampler::LDFCompatibleSampler,
    initial_params::Union{AbstractVector,Nothing}=nothing,
    # We could set `link=requires_unconstrained_space(sampler)`, but that would
    # preclude moving `initialise_varinfo` to DynamicPPL, since
    # `requires_unconstrained_space` is defined in Turing (unless that function
    # is also moved to DynamicPPL, or AbstractMCMC)
    link::Bool=false,
)
    init_sampler = DynamicPPL.initialsampler(sampler)
    vi = DynamicPPL.typed_varinfo(rng, model, init_sampler)

    # Update the parameters if provided.
    if initial_params !== nothing
        # Note that initialize_parameters!! expects parameters in to be
        # specified in unconstrained space. TODO: Make this more generic.
        vi = DynamicPPL.initialize_parameters!!(vi, initial_params, model)

        # Update joint log probability.
        # This is a quick fix for https://github.com/TuringLang/Turing.jl/issues/1588
        # and https://github.com/TuringLang/Turing.jl/issues/1563
        # to avoid that existing variables are resampled
        vi = last(DynamicPPL.evaluate!!(model, vi, DynamicPPL.DefaultContext()))
    end

    if link
        vi = DynamicPPL.link(vi, model)
    end

    return vi
end

####################################################
### The rest of this file is boring boilerplate. ###
####################################################

function AbstractMCMC.sample(
    model_or_ldf::Union{Model,LogDensityFunction},
    alg_or_spl::Union{LDFCompatibleAlgorithm,Sampler{<:LDFCompatibleAlgorithm}},
    N::Integer;
    kwargs...,
)
    return AbstractMCMC.sample(Random.default_rng(), model_or_ldf, alg_or_spl, N; kwargs...)
end

function AbstractMCMC.sample(
    rng::Random.AbstractRNG,
    ldf::LogDensityFunction,
    alg::LDFCompatibleAlgorithm,
    N::Integer;
    kwargs...,
)
    return AbstractMCMC.sample(rng, ldf, Sampler(alg), N; kwargs...)
end

function AbstractMCMC.sample(
    rng::Random.AbstractRNG,
    model::Model,
    alg::LDFCompatibleAlgorithm,
    N::Integer;
    kwargs...,
)
    return AbstractMCMC.sample(rng, model, Sampler(alg), N; kwargs...)
end

function AbstractMCMC.sample(
    rng::Random.AbstractRNG,
    model::Model,
    spl::Sampler{<:LDFCompatibleAlgorithm},
    N::Integer;
    kwargs...,
)
    initial_params = get(kwargs, :initial_params, nothing)
    link = requires_unconstrained_space(spl)
    vi = initialise_varinfo(rng, model, spl, initial_params, link)
    ldf = LogDensityFunction(model, vi; adtype=get_adtype(spl))
    return AbstractMCMC.sample(rng, ldf, spl, N; kwargs...)
end

function AbstractMCMC.sample(
    model_or_ldf::Union{Model,LogDensityFunction},
    alg_or_spl::Union{LDFCompatibleAlgorithm,Sampler{<:LDFCompatibleAlgorithm}},
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
    alg::LDFCompatibleAlgorithm,
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
    alg::LDFCompatibleAlgorithm,
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
    spl::LDFCompatibleSampler,
    ensemble::AbstractMCMC.AbstractMCMCEnsemble,
    N::Integer,
    n_chains::Integer;
    kwargs...,
)
    initial_params = get(kwargs, :initial_params, nothing)
    link = requires_unconstrained_space(spl)
    vi = initialise_varinfo(rng, model, spl, initial_params, link)
    ldf = LogDensityFunction(model, vi; adtype=get_adtype(spl))
    return AbstractMCMC.sample(rng, ldf, spl, ensemble, N, n_chains; kwargs...)
end

end # module
