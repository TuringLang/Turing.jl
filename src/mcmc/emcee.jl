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
struct Emcee{E<:AMH.Ensemble} <: AbstractSampler
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

struct EmceeState{L<:LogDensityFunction,S}
    ldf::L
    states::S
end

# Utility function to tetrieve the number of walkers
_get_n_walkers(e::Emcee) = e.ensemble.n_walkers

# Because Emcee expects n_walkers initialisations, we need to override this
function Turing.Inference.init_strategy(spl::Emcee)
    return fill(DynamicPPL.InitFromPrior(), _get_n_walkers(spl))
end
# We also have to explicitly allow this or else it will error...
function Turing.Inference._convert_initial_params(
    x::AbstractVector{<:DynamicPPL.AbstractInitStrategy}
)
    return x
end

function AbstractMCMC.step(
    rng::Random.AbstractRNG, model::Model, spl::Emcee; initial_params, kwargs...
)
    # Sample from the prior
    n = _get_n_walkers(spl)
    vis = [VarInfo(rng, model) for _ in 1:n]

    # Update the parameters if provided.
    if !(
        initial_params isa AbstractVector{<:DynamicPPL.AbstractInitStrategy} &&
        length(initial_params) == n
    )
        err_msg = "initial_params for `Emcee` must be a vector of `DynamicPPL.AbstractInitStrategy`, with length equal to the number of walkers ($n)"
        throw(ArgumentError(err_msg))
    end
    vis = map(vis, initial_params) do vi, strategy
        last(DynamicPPL.init!!(rng, model, vi, strategy))
    end

    # Compute initial transition and states.
    transition = [DynamicPPL.ParamsWithStats(vi, model) for vi in vis]

    linked_vi = DynamicPPL.link!!(vis[1], model)
    state = EmceeState(
        DynamicPPL.LogDensityFunction(model, getlogjoint_internal, linked_vi),
        map(vis) do vi
            vi = DynamicPPL.link!!(vi, model)
            AMH.Transition(vi[:], DynamicPPL.getlogjoint_internal(vi), false)
        end,
    )

    return transition, state
end

function AbstractMCMC.step(
    rng::AbstractRNG, model::Model, spl::Emcee, state::EmceeState; kwargs...
)
    # Generate a log joint function.
    densitymodel = AMH.DensityModel(Base.Fix1(LogDensityProblems.logdensity, state.ldf))

    # Compute the next states.
    _, states = AbstractMCMC.step(rng, densitymodel, spl.ensemble, state.states)

    # Compute the next transition and state.
    transition = map(states) do _state
        return DynamicPPL.ParamsWithStats(
            _state.params, state.ldf, AbstractMCMC.getstats(_state)
        )
    end
    newstate = EmceeState(state.ldf, states)

    return transition, newstate
end

function AbstractMCMC.bundle_samples(
    samples::Vector{<:Vector},
    model::AbstractModel,
    spl::Emcee,
    state::EmceeState,
    chain_type::Type{MCMCChains.Chains};
    kwargs...,
)
    n_walkers = _get_n_walkers(spl)
    chains = map(1:n_walkers) do i
        this_walker_samples = [s[i] for s in samples]
        AbstractMCMC.bundle_samples(
            this_walker_samples, model, spl, state, chain_type; kwargs...
        )
    end
    return AbstractMCMC.chainscat(chains...)
end
