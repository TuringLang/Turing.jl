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
function Turing._convert_initial_params(
    x::AbstractVector{<:DynamicPPL.AbstractInitStrategy}
)
    return x
end

function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::Model,
    spl::Emcee;
    initial_params,
    discard_sample=false,
    kwargs...,
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
        last(DynamicPPL.init!!(rng, model, vi, strategy, DynamicPPL.UnlinkAll()))
    end

    # Compute initial transition and states.
    transition = if discard_sample
        nothing
    else
        [
            DynamicPPL.ParamsWithStats(
                DynamicPPL.InitFromParams(DynamicPPL.get_values(vi)), model
            ) for vi in vis
        ]
    end

    linked_vis = map(vi -> DynamicPPL.link!!(vi, model), vis)
    check_walkers_same_dimension(linked_vis)
    state = EmceeState(
        DynamicPPL.LogDensityFunction(model, getlogjoint_internal, linked_vis[1]),
        map(linked_vis) do vi
            AMH.Transition(vi[:], DynamicPPL.getlogjoint_internal(vi), false)
        end,
    )

    return transition, state
end

"""
    check_walkers_same_dimension(linked_vis)

Throw unless every walker has the same number of parameters.

The stretch move interpolates between two walkers' position vectors, so the ensemble needs one
parameter space shared by all of them. On a model whose set of variables depends on its own
draws, walkers initialised from the prior can land in different branches and so have different
lengths, and the mismatch otherwise surfaces as a `DimensionMismatch` from inside the proposal's
broadcast.

The check is necessary rather than sufficient: it rules out walkers that start in different
branches, not a model whose dimension varies at all. A proposal can still reach a vector whose
trace visits other variables, and the single `LogDensityFunction` built here would decode it
against the layout of the first walker. `Emcee` has no way to express that, so a dynamic model
is best avoided with it entirely.
"""
function check_walkers_same_dimension(linked_vis)
    dims = map(vi -> length(vi[:]), linked_vis)
    allequal(dims) && return nothing
    throw(
        ArgumentError(
            "`Emcee`'s walkers have different numbers of parameters ($(join(sort(unique(dims)), ", "))). " *
            "The stretch move moves along the line between two walkers, so they all have to " *
            "live in one parameter space. This model's set of variables depends on its own " *
            "draws, which `Emcee` cannot sample.",
        ),
    )
end

function AbstractMCMC.step(
    rng::AbstractRNG,
    model::Model,
    spl::Emcee,
    state::EmceeState;
    discard_sample=false,
    kwargs...,
)
    # Generate a log joint function.
    densitymodel = AMH.DensityModel(Base.Fix1(LogDensityProblems.logdensity, state.ldf))

    # Compute the next states.
    _, states = AbstractMCMC.step(rng, densitymodel, spl.ensemble, state.states)

    # Compute the next transition and state.
    transition = if discard_sample
        nothing
    else
        map(states) do _state
            return DynamicPPL.ParamsWithStats(
                _state.params, state.ldf, AbstractMCMC.getstats(_state)
            )
        end
    end
    newstate = EmceeState(state.ldf, states)

    return transition, newstate
end

function AbstractMCMC.bundle_samples(
    samples::Vector{<:AbstractVector},
    model::DynamicPPL.Model,
    spl::Emcee,
    state::EmceeState,
    chain_type::Type{VNChain};
    kwargs...,
)
    n_walkers = _get_n_walkers(spl)
    chains = map(1:n_walkers) do i
        this_walker_samples = [s[i] for s in samples]
        AbstractMCMC.bundle_samples(
            this_walker_samples, model, spl, state, VNChain; kwargs...
        )
    end
    return AbstractMCMC.chainscat(chains...)
end
