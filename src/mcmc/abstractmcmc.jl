# TODO: Implement additional checks for certain samplers, e.g.
# HMC not supporting discrete parameters.
function _check_model(model::DynamicPPL.Model)
    new_model = DynamicPPL.setleafcontext(model, DynamicPPL.InitContext())
    return DynamicPPL.check_model(
        new_model, DynamicPPL.OnlyAccsVarInfo(); error_on_failure=true
    )
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
        initial_params=Turing._convert_initial_params(initial_params),
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
        check_model=false, # no need to check again
        initial_params=map(Turing._convert_initial_params, initial_params),
        kwargs...,
    )
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
