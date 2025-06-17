# This file contains the basic methods for `AbstractMCMC.sample`.
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
# evaluated (only a LogDensityFunction has that information). The methods defined
# in this file provide the 'bridge' between these two, and also provide hooks to
# allow for some special behaviour, e.g. setting the default chain type to
# MCMCChains.Chains, and also checking the model with DynamicPPL.check_model.
#
# Advanced users who want to customise the way their model is executed (e.g. by
# using different types of VarInfo) can construct their own LogDensityFunction
# and call `sample(ldf, spl, N)` themselves.

# Because this is a pain to implement all at once, we do it for one sampler at a time.
# This type tells us which samplers have been 'updated' to the new interface.
const LDFCompatibleSampler = Union{Hamiltonian,ESS,MH}

"""
    sample(
        [rng::Random.AbstractRNG, ]
        model::DynamicPPL.Model,
        alg::InferenceAlgorithm,
        N::Integer;
        kwargs...
    )
    sample(
        [rng::Random.AbstractRNG, ]
        ldf::DynamicPPL.LogDensityFunction,
        alg::InferenceAlgorithm,
        N::Integer;
        kwargs...
    )

Perform MCMC sampling on the given `model` or `ldf` using the specified `alg`,
for `N` iterations.

If a `DynamicPPL.Model` is passed as the `model` argument, it will be converted
into a `DynamicPPL.LogDensityFunction` internally, which is then used for
sampling. If necessary, the AD backend used for sampling will be inferred from
the sampler.

A `LogDensityFunction` contains both a model as well as a `VarInfo` object. In
the case where a `DynamicPPL.Model` is passed, the associated `varinfo` is
created using the `initialise_varinfo` function; by default, this generates a
`DynamicPPL.VarInfo{<:NamedTuple}` object (i.e. a 'typed VarInfo'). If you need
to customise the type of VarInfo used during sampling, you can construct a
`LogDensityFunction` yourself and pass it to this method.

If you are passing an `ldf::LogDensityFunction` to a gradient-based sampler,
`ldf.adtype` must be set to an `AbstractADType` (using the constructor
`LogDensityFunction(model, varinfo; adtype=adtype)`). Any `adtype` information
in the sampler will be ignored, in favour of the one in the `ldf`.

For a list of typical keyword arguments to `sample`, please see
https://turinglang.org/AbstractMCMC.jl/stable/api/#Common-keyword-arguments.
"""
function AbstractMCMC.sample(
    rng::Random.AbstractRNG,
    ldf::LogDensityFunction,
    spl::LDFCompatibleSampler,
    N::Integer;
    check_model::Bool=true,
    chain_type=MCMCChains.Chains,
    progress=PROGRESS[],
    resume_from=nothing,
    initial_state=DynamicPPL.loadstate(resume_from),
    kwargs...,
)
    # LDF needs to be set with SamplingContext, or else samplers cannot
    # overload the tilde-pipeline.
    ctx = if ldf.context isa SamplingContext
        ldf.context
    else
        SamplingContext(rng, spl, ldf.context)
    end
    # Note that, in particular, sampling can mutate the variables in the LDF's
    # varinfo (because it ultimately ends up calling `evaluate!!(ldf.model,
    # ldf.varinfo)`. Furthermore, the first call to `AbstractMCMC.step` assumes
    # that the parameters in the LDF are the initial parameters. So, we need to
    # deepcopy the varinfo here to ensure that sample(rng, ldf, ...) is
    # reproducible.
    vi = deepcopy(ldf.varinfo)
    # TODO(penelopeysm): Unsure if model needes to be deepcopied as well.
    # Note that deepcopying the entire LDF is risky as it may include e.g.
    # Mooncake or Enzyme types that don't deepcopy well. I ran into an issue
    # where Mooncake errored when deepcopying an LDF.
    ldf = LogDensityFunction(ldf.model, vi, ctx; adtype=ldf.adtype)
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
"""
    sample(
        [rng::Random.AbstractRNG, ]
        model::DynamicPPL.Model,
        alg::InferenceAlgorithm,
        ensemble::AbstractMCMC.AbstractMCMCEnsemble,
        N::Integer;
        n_chains::Integer;
        kwargs...
    )
    sample(
        [rng::Random.AbstractRNG, ]
        ldf::DynamicPPL.LogDensityFunction,
        alg::InferenceAlgorithm,
        ensemble::AbstractMCMC.AbstractMCMCEnsemble,
        N::Integer;
        n_chains::Integer;
        kwargs...
    )

Sample from the given `model` or `ldf` using the specified `alg`, for `N`
iterations per chain, with `n_chains` chains in total. The `ensemble` argument
specifies how sampling is to be carried out: this can be `MCMCSerial` for
serial (i.e. single-threaded, sequential) sampling, `MCMCThreads` for sampling
using Julia's threads, or `MCMCDistributed` for distributed sampling across
multiple processes.

All other arguments are the same as in `sample([rng, ]model, alg, N; kwargs...)`.
"""
function AbstractMCMC.sample(
    rng::Random.AbstractRNG,
    ldf::LogDensityFunction,
    spl::LDFCompatibleSampler,
    ensemble::AbstractMCMC.AbstractMCMCEnsemble,
    N::Integer,
    n_chains::Integer;
    check_model::Bool=true,
    chain_type=MCMCChains.Chains,
    progress=PROGRESS[],
    resume_from=nothing,
    initial_state=DynamicPPL.loadstate(resume_from),
    kwargs...,
)
    # LDF needs to be set with SamplingContext, or else samplers cannot
    # overload the tilde-pipeline.
    ctx = if ldf.context isa SamplingContext
        ldf.context
    else
        SamplingContext(rng, spl, ldf.context)
    end
    # Note that, in particular, sampling can mutate the variables in the LDF's
    # varinfo (because it ultimately ends up calling `evaluate!!(ldf.model,
    # ldf.varinfo)`. Furthermore, the first call to `AbstractMCMC.step` assumes
    # that the parameters in the LDF are the initial parameters. So, we need to
    # deepcopy the varinfo here to ensure that sample(rng, ldf, ...) is
    # reproducible.
    vi = deepcopy(ldf.varinfo)
    # TODO(penelopeysm): Unsure if model needes to be deepcopied as well.
    # Note that deepcopying the entire LDF is risky as it may include e.g.
    # Mooncake or Enzyme types that don't deepcopy well. I ran into an issue
    # where Mooncake errored when deepcopying an LDF.
    ldf = LogDensityFunction(ldf.model, vi, ctx; adtype=ldf.adtype)
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
        new_kwargs...,
    )
end

# This method should be in DynamicPPL. We will move it there when all the
# Turing samplers have been updated.
"""
    initialise_varinfo(rng, model, sampler, initial_params=nothing, link=false)

Return a suitable initial varinfo object, which will be used when sampling
`model` with `sampler`. If given, the initial parameter values will be set in
the varinfo object. Also performs linking if requested.

# Arguments
- `rng::Random.AbstractRNG`: Random number generator.
- `model::Model`: Model for which we want to create a varinfo object.
- `sampler::AbstractSampler`: Sampler which will make use of the varinfo object.
- `initial_params::Union{AbstractVector,Nothing}`: Initial parameter values to
be set in the varinfo object. Note that these should be given in unconstrained
space.
- `link::Bool`: Whether to link the varinfo.

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

    return if link
        DynamicPPL.link(vi, model)
    else
        vi
    end
end

##########################################################################
### Everything below this is boring boilerplate for the new interface. ###
##########################################################################

function AbstractMCMC.sample(model::Model, spl::LDFCompatibleSampler, N::Integer; kwargs...)
    return AbstractMCMC.sample(Random.default_rng(), model, spl, N; kwargs...)
end

function AbstractMCMC.sample(
    ldf::LogDensityFunction, spl::LDFCompatibleSampler, N::Integer; kwargs...
)
    return AbstractMCMC.sample(Random.default_rng(), ldf, spl, N; kwargs...)
end

function AbstractMCMC.sample(
    rng::Random.AbstractRNG,
    model::Model,
    spl::LDFCompatibleSampler,
    N::Integer;
    check_model::Bool=true,
    kwargs...,
)
    # Annoying: Need to run check_model before initialise_varinfo so that
    # errors in the model are caught gracefully (as initialise_varinfo also
    # runs the model and will throw ugly errors if the model is incorrect).
    check_model && DynamicPPL.check_model(model; error_on_failure=true)
    initial_params = get(kwargs, :initial_params, nothing)
    link = requires_unconstrained_space(spl)
    vi = initialise_varinfo(rng, model, spl, initial_params, link)
    ctx = SamplingContext(rng, spl, model.context)
    ldf = LogDensityFunction(model, vi, ctx; adtype=get_adtype(spl))
    # No need to run check_model again
    return AbstractMCMC.sample(rng, ldf, spl, N; kwargs..., check_model=false)
end

function AbstractMCMC.sample(
    model::Model,
    spl::LDFCompatibleSampler,
    ensemble::AbstractMCMC.AbstractMCMCEnsemble,
    N::Integer,
    n_chains::Integer;
    kwargs...,
)
    return AbstractMCMC.sample(
        Random.default_rng(), model, spl, ensemble, N, n_chains; kwargs...
    )
end

function AbstractMCMC.sample(
    ldf::LogDensityFunction,
    spl::LDFCompatibleSampler,
    ensemble::AbstractMCMC.AbstractMCMCEnsemble,
    N::Integer,
    n_chains::Integer;
    kwargs...,
)
    return AbstractMCMC.sample(
        Random.default_rng(), ldf, spl, ensemble, N, n_chains; kwargs...
    )
end

function AbstractMCMC.sample(
    rng::Random.AbstractRNG,
    model::Model,
    spl::LDFCompatibleSampler,
    ensemble::AbstractMCMC.AbstractMCMCEnsemble,
    N::Integer,
    n_chains::Integer;
    check_model::Bool=true,
    kwargs...,
)
    # Annoying: Need to run check_model before initialise_varinfo so that
    # errors in the model are caught gracefully (as initialise_varinfo also
    # runs the model and will throw ugly errors if the model is incorrect).
    check_model && DynamicPPL.check_model(model; error_on_failure=true)
    initial_params = get(kwargs, :initial_params, nothing)
    link = requires_unconstrained_space(spl)
    vi = initialise_varinfo(rng, model, spl, initial_params, link)
    ctx = SamplingContext(rng, spl, model.context)
    ldf = LogDensityFunction(model, vi, ctx; adtype=get_adtype(spl))
    # No need to run check_model again
    return AbstractMCMC.sample(
        rng, ldf, spl, ensemble, N, n_chains; kwargs..., check_model=false
    )
end

########################################################
# DEPRECATED SAMPLE METHODS                            #
########################################################
# All the code below should eventually be removed.     #
# We need to keep it here for now so that the          #
# inference algorithms that _haven't_ yet been updated #
# to take LogDensityFunction still work.               #
########################################################

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
    check_model && DynamicPPL.check_model(model)
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
    check_model && DynamicPPL.check_model(model)
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
