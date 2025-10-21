# TODO: Implement additional checks for certain samplers, e.g.
# HMC not supporting discrete parameters.
function _check_model(model::DynamicPPL.Model)
    new_model = DynamicPPL.setleafcontext(model, DynamicPPL.InitContext())
    return DynamicPPL.check_model(new_model, VarInfo(); error_on_failure=true)
end
function _check_model(model::DynamicPPL.Model, ::AbstractSampler)
    return _check_model(model)
end

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
    default_varinfo(rng, model, sampler)

Return a default varinfo object for the given `model` and `sampler`.
The default method for this returns a NTVarInfo (i.e. 'typed varinfo').
"""
function default_varinfo(
    rng::Random.AbstractRNG, model::DynamicPPL.Model, ::AbstractSampler
)
    # Note that in `AbstractMCMC.step`, the values in the varinfo returned here are
    # immediately overwritten by a subsequent call to `init!!`. The reason why we
    # _do_ create a varinfo with parameters here (as opposed to simply returning
    # an empty `typed_varinfo(VarInfo())`) is to avoid issues where pushing to an empty
    # typed VarInfo would fail. This can happen if two VarNames have different types
    # but share the same symbol (e.g. `x.a` and `x.b`).
    # TODO(mhauru) Fix push!! to work with arbitrary lens types, and then remove the arguments
    # and return an empty VarInfo instead.
    return DynamicPPL.typed_varinfo(VarInfo(rng, model))
end

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
    kwargs...,
)
    check_model && _check_model(model, spl)
    return AbstractMCMC.mcmcsample(
        rng,
        model,
        spl,
        N;
        initial_params=_convert_initial_params(initial_params),
        chain_type,
        kwargs...,
    )
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
    initial_params=fill(init_strategy(spl), n_chains),
    kwargs...,
)
    check_model && _check_model(model, spl)
    if !(initial_params isa AbstractVector) || length(initial_params) != n_chains
        errmsg = "`initial_params` must be an AbstractVector of length `n_chains`; one element per chain"
        throw(ArgumentError(errmsg))
    end
    return AbstractMCMC.mcmcsample(
        rng,
        model,
        spl,
        ensemble,
        N,
        n_chains;
        chain_type,
        initial_params=map(_convert_initial_params, initial_params),
        kwargs...,
    )
end

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
    # Generate the default varinfo. Note that any parameters inside this varinfo
    # will be immediately overwritten by the next call to `init!!`.
    vi = default_varinfo(rng, model, spl)

    # Fill it with initial parameters. Note that, if `InitFromParams` is used, the
    # parameters provided must be in unlinked space (when inserted into the
    # varinfo, they will be adjusted to match the linking status of the
    # varinfo).
    _, vi = DynamicPPL.init!!(rng, model, vi, initial_params)

    # Call the actual function that does the first step.
    return initialstep(rng, model, spl, vi; initial_params, kwargs...)
end
