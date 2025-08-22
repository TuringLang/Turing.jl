###
### Sampler states
###

"""
    Emcee(n_walkers::Int, stretch_length=2.0)

Affine-invariant ensemble sampling algorithm.

# Reference

Foreman-Mackey, D., Hogg, D. W., Lang, D., & Goodman, J. (2013).
emcee: The MCMC Hammer. Publications of the Astronomical Society of the
Pacific, 125 (925), 306. https://doi.org/10.1086/670067
"""
struct Emcee{E<:AMH.Ensemble} <: InferenceAlgorithm
    ensemble::E
end

function Emcee(n_walkers::Int, stretch_length=2.0)
    # Note that the proposal distribution here is just a Normal(0,1)
    # because we do not need AdvancedMH to know the proposal for
    # ensemble sampling.
    prop = AMH.StretchProposal(nothing, stretch_length)
    ensemble = AMH.Ensemble(n_walkers, prop)
    return Emcee{typeof(ensemble)}(ensemble)
end

struct EmceeState{V<:AbstractVarInfo,S}
    vi::V
    states::S
end

function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::Model,
    spl::Sampler{<:Emcee};
    resume_from=nothing,
    initial_params=nothing,
    kwargs...,
)
    if resume_from !== nothing
        state = loadstate(resume_from)
        return AbstractMCMC.step(rng, model, spl, state; kwargs...)
    end

    # Sample from the prior
    n = spl.alg.ensemble.n_walkers
    vis = [VarInfo(rng, model, SampleFromPrior()) for _ in 1:n]

    # Update the parameters if provided.
    if initial_params !== nothing
        length(initial_params) == n ||
            throw(ArgumentError("initial parameters have to be specified for each walker"))
        vis = map(vis, initial_params) do vi, init
            # TODO(DPPL0.38/penelopeysm) This whole thing can be replaced with init!!
            vi = DynamicPPL.initialize_parameters!!(vi, init, model)

            # Update log joint probability.
            spl_model = DynamicPPL.contextualize(
                model, DynamicPPL.SamplingContext(rng, SampleFromPrior(), model.context)
            )
            last(DynamicPPL.evaluate!!(spl_model, vi))
        end
    end

    # Compute initial transition and states.
    transition = [Transition(model, vi, nothing) for vi in vis]

    # TODO: Make compatible with immutable `AbstractVarInfo`.
    state = EmceeState(
        vis[1],
        map(vis) do vi
            vi = DynamicPPL.link!!(vi, model)
            AMH.Transition(vi[:], DynamicPPL.getlogjoint_internal(vi), false)
        end,
    )

    return transition, state
end

function AbstractMCMC.step(
    rng::AbstractRNG, model::Model, spl::Sampler{<:Emcee}, state::EmceeState; kwargs...
)
    # Generate a log joint function.
    vi = state.vi
    densitymodel = AMH.DensityModel(
        Base.Fix1(
            LogDensityProblems.logdensity,
            DynamicPPL.LogDensityFunction(model, DynamicPPL.getlogjoint_internal, vi),
        ),
    )

    # Compute the next states.
    t, states = AbstractMCMC.step(rng, densitymodel, spl.alg.ensemble, state.states)

    # Compute the next transition and state.
    transition = map(states) do _state
        vi = DynamicPPL.unflatten(vi, _state.params)
        return Transition(model, vi, t)
    end
    newstate = EmceeState(vi, states)

    return transition, newstate
end

function AbstractMCMC.bundle_samples(
    samples::Vector{<:Vector},
    model::AbstractModel,
    spl::Sampler{<:Emcee},
    state::EmceeState,
    chain_type::Type{MCMCChains.Chains};
    save_state=false,
    sort_chain=false,
    discard_initial=0,
    thinning=1,
    kwargs...,
)
    # Convert transitions to array format.
    # Also retrieve the variable names.
    params_vec = map(Base.Fix1(_params_to_array, model), samples)

    # Extract names and values separately.
    varnames = params_vec[1][1]
    varnames_symbol = map(Symbol, varnames)
    vals_vec = [p[2] for p in params_vec]

    # Get the values of the extra parameters in each transition.
    extra_vec = map(get_transition_extras, samples)

    # Get the extra parameter names & values.
    extra_params = extra_vec[1][1]
    extra_values_vec = [e[2] for e in extra_vec]

    # Extract names & construct param array.
    nms = [varnames_symbol; extra_params]
    # `hcat` first to ensure we get the right `eltype`.
    x = hcat(first(vals_vec), first(extra_values_vec))
    # Pre-allocate to minimize memory usage.
    parray = Array{eltype(x),3}(undef, length(vals_vec), size(x, 2), size(x, 1))
    for (i, (vals, extras)) in enumerate(zip(vals_vec, extra_values_vec))
        parray[i, :, :] = transpose(hcat(vals, extras))
    end

    # Get the average or final log evidence, if it exists.
    le = getlogevidence(samples, state, spl)

    # Set up the info tuple.
    info = (varname_to_symbol=OrderedDict(zip(varnames, varnames_symbol)),)
    if save_state
        info = merge(info, (model=model, sampler=spl, samplerstate=state))
    end

    # Concretize the array before giving it to MCMCChains.
    parray = MCMCChains.concretize(parray)

    # Chain construction.
    chain = MCMCChains.Chains(
        parray,
        nms,
        (internals=extra_params,);
        evidence=le,
        info=info,
        start=discard_initial + 1,
        thin=thinning,
    )

    return sort_chain ? sort(chain) : chain
end
