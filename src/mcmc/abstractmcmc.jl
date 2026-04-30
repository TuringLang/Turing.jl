"""
    Turing.Inference.init_strategy(spl::AbstractSampler)

Get the default initialization strategy for a given sampler `spl`, i.e. how initial
parameters for sampling are chosen if not specified by the user. By default, this is
`InitFromPrior()`, which samples initial parameters from the prior distribution.
"""
init_strategy(::AbstractSampler) = DynamicPPL.InitFromPrior()

"""
    _convert_initial_params(initial_params)

Convert `initial_params` to a `DynamicPPl.AbstractInitStrategy` if it is not already one, or
throw a useful error message.
"""
_convert_initial_params(initial_params::DynamicPPL.AbstractInitStrategy) = initial_params
function _convert_initial_params(nt::NamedTuple)
    @info "Using a NamedTuple for `initial_params` will be deprecated in a future release. Please use `InitFromParams(namedtuple)` instead."
    return DynamicPPL.InitFromParams(nt)
end
function _convert_initial_params(d::AbstractDict{<:VarName})
    @info "Using a Dict for `initial_params` will be deprecated in a future release. Please use `InitFromParams(dict)` instead."
    return DynamicPPL.InitFromParams(d)
end
function _convert_initial_params(::AbstractVector{<:Real})
    errmsg = "`initial_params` must be a `NamedTuple`, an `AbstractDict{<:VarName}`, or ideally a `DynamicPPL.AbstractInitStrategy`. Using a vector of parameters for `initial_params` is no longer supported. Please see https://turinglang.org/docs/usage/sampling-options/#specifying-initial-parameters for details on how to update your code."
    throw(ArgumentError(errmsg))
end
function _convert_initial_params(@nospecialize(_::Any))
    errmsg = "`initial_params` must be a `NamedTuple`, an `AbstractDict{<:VarName}`, or a `DynamicPPL.AbstractInitStrategy`."
    throw(ArgumentError(errmsg))
end

"""
    find_initial_params_ldf(rng, ldf, init_strategy; max_attempts=1000)

Given a `LogDensityFunction` and an initialization strategy, attempt to find valid initial
parameters by sampling from the initialization strategy and checking that the log density
(and gradient, if available) are finite. If valid parameters are not found after
`max_attempts`, throw an error.
"""
function find_initial_params_ldf(
    rng::Random.AbstractRNG,
    ldf::DynamicPPL.LogDensityFunction,
    init_strategy::DynamicPPL.AbstractInitStrategy;
    max_attempts::Int=1000,
)
    for attempts in 1:max_attempts
        # Get new parameters
        x = rand(rng, ldf, init_strategy)
        is_valid = if ldf.adtype === nothing
            logp = LogDensityProblems.logdensity(ldf, x)
            isfinite(logp)
        else
            logp, grad = LogDensityProblems.logdensity_and_gradient(ldf, x)
            isfinite(logp) && all(isfinite, grad)
        end

        # If they're OK, return them
        is_valid && return x

        attempts == 10 &&
            @warn "failed to find valid initial parameters in $(attempts) tries; consider providing a different initialisation strategy with the `initial_params` keyword"
    end

    # if we failed to find valid initial parameters, error
    return error(
        "failed to find valid initial parameters in $(max_attempts) tries. See https://turinglang.org/docs/uri/initial-parameters for common causes and solutions. If the issue persists, please open an issue at https://github.com/TuringLang/Turing.jl/issues",
    )
end

"""
    post_sample_hook(chain, sampler::AbstractSampler; kwargs...)

A post-sampling hook that can e.g. print info about the results of sampling.

Implementations of this should be careful to take `kwargs...` as keyword arguments instead
of restricting the signature to specific keyword arguments. Right now, the only keyword
argument that is passed to this function is `verbose`, but in the future additional keyword
arguments may be passed here.
"""
post_sample_hook(chain, ::AbstractSampler; kwargs...) = nothing

#########################################
# Default definitions for the interface #
#########################################

function AbstractMCMC.sample(
    model::DynamicPPL.Model, spl::AbstractSampler, N::Integer; kwargs...
)
    return AbstractMCMC.sample(Random.default_rng(), model, spl, N; kwargs...)
end

function AbstractMCMC.sample(
    rng::AbstractRNG,
    model::DynamicPPL.Model,
    spl::AbstractSampler,
    N::Integer;
    initial_params=init_strategy(spl),
    check_model::Bool=true,
    chain_type=DEFAULT_CHAIN_TYPE,
    verbose::Bool=true,
    kwargs...,
)
    check_model && Turing._check_model(model, spl)
    chain = AbstractMCMC.mcmcsample(
        rng,
        model,
        spl,
        N;
        initial_params=Turing._convert_initial_params(initial_params),
        chain_type,
        verbose,
        kwargs...,
    )
    post_sample_hook(chain, spl; verbose)
    return chain
end

function AbstractMCMC.sample(
    model::DynamicPPL.Model,
    alg::AbstractSampler,
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
    model::DynamicPPL.Model,
    spl::AbstractSampler,
    ensemble::AbstractMCMC.AbstractMCMCEnsemble,
    N::Integer,
    n_chains::Integer;
    chain_type=DEFAULT_CHAIN_TYPE,
    check_model::Bool=true,
    verbose::Bool=true,
    initial_params=fill(init_strategy(spl), n_chains),
    kwargs...,
)
    check_model && Turing._check_model(model, spl)
    if !(initial_params isa AbstractVector) || length(initial_params) != n_chains
        errmsg = "`initial_params` must be an AbstractVector of length `n_chains`; one element per chain"
        throw(ArgumentError(errmsg))
    end
    chain = AbstractMCMC.mcmcsample(
        rng,
        model,
        spl,
        ensemble,
        N,
        n_chains;
        chain_type,
        check_model=false, # no need to check again
        initial_params=map(Turing._convert_initial_params, initial_params),
        verbose,
        kwargs...,
    )
    post_sample_hook(chain, spl; verbose)
    return chain
end

"""
    loadstate(chain::MCMCChains.Chains)

Load the final state of the sampler from a `MCMCChains.Chains` object.

To save the final state of the sampler, you must use `sample(...; save_state=true)`. If this
argument was not used during sampling, calling `loadstate` will throw an error.
"""
function loadstate(chain::MCMCChains.Chains)
    if !haskey(chain.info, :samplerstate)
        throw(
            ArgumentError(
                "the chain object does not contain the final state of the sampler; to save the final state you must sample with `save_state=true`",
            ),
        )
    end
    return chain.info[:samplerstate]
end

# TODO(penelopeysm): Remove initialstep and generalise MCMC sampling procedures
function initialstep end

function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::DynamicPPL.Model,
    spl::AbstractSampler;
    initial_params,
    kwargs...,
)
    # Generate a VarInfo with initial parameters. Note that, if `InitFromParams` is used,
    # the parameters provided must be in unlinked space (when inserted into the varinfo,
    # they will be adjusted to match the linking status of the varinfo).
    _, vi = DynamicPPL.init!!(rng, model, VarInfo(), initial_params, DynamicPPL.UnlinkAll())

    # Call the actual function that does the first step.
    return initialstep(rng, model, spl, vi; initial_params, kwargs...)
end
