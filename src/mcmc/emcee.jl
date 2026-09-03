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
    check_walkers_same_layout(linked_vis)
    state = EmceeState(
        DynamicPPL.LogDensityFunction(model, getlogjoint_internal, linked_vis[1]),
        map(linked_vis) do vi
            AMH.Transition(vi[:], DynamicPPL.getlogjoint_internal(vi), false)
        end,
    )

    return transition, state
end

"""
    check_walkers_same_layout(linked_vis)

Throw unless every walker occupies the same parameter layout.

The stretch move interpolates between two walkers' position vectors, and the single
`LogDensityFunction` is built from the first walker, so every walker's vector is decoded against
that one layout. All of them therefore have to hold the same variables in the same order at the
same widths. On a model whose set of variables, or their sizes, depends on its own draws,
walkers initialised from the prior do not.

Each variable's name, order, and width are compared, rather than only the names or the total
width. Comparing names alone passed walkers agreeing on names while a variable's dimension
differed, and names with the total width passed two variables trading dimensions -- `x` of 2 and
`y` of 3 against `x` of 3 and `y` of 2 -- both of which then failed inside the decode or proposal.

Widths and not transforms: the layout fixes each variable's range, while its link is re-derived
at every evaluation, so walkers holding one variable under different transforms still sample
correctly, and comparing transforms would refuse them.

This is necessary rather than sufficient, and only the *initial* walkers are examined. A
proposal can still cross into a branch none of them started in, and the decode then fails on a
variable the layout has no range for -- `m ~ Normal(); m > 0 ? (x ~ Normal()) : (y ~ Normal())`,
started entirely in `m > 0`, raises `KeyError: key y not found` once a proposal reaches `m < 0`.
Catching that would mean validating every evaluation. A model whose layout varies at all is best
not sampled with `Emcee`.
"""
function check_walkers_same_layout(linked_vis)
    layouts = map(linked_vis) do vi
        [
            vn => length(DynamicPPL.get_internal_value(tv)) for
            (vn, tv) in pairs(DynamicPPL.get_values(vi))
        ]
    end
    allequal(layouts) && return nothing
    shown = unique(
        map(l -> string("[", join(("$k of $v" for (k, v) in l), ", "), "]"), layouts)
    )
    return throw(
        ArgumentError(
            "`Emcee`'s walkers do not occupy the same parameter layout " *
            "($(join(shown, " versus "))). The stretch move moves along the line between two " *
            "walkers and every walker is decoded against one layout, so they all have to live " *
            "in one parameter space. This model's variables, or their sizes, depend on its own " *
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
