###
### Gibbs samplers / compositional samplers.
###


"""
    isgibbscomponent(alg)

Determine whether algorithm `alg` is allowed as a Gibbs component.
"""
isgibbscomponent(alg) = false


"""
    Gibbs(algs...)

Compositional MCMC interface. Gibbs sampling combines one or more
sampling algorithms, each of which samples from a different set of
variables in a model.

Example:
```julia
@model gibbs_example(x) = begin
    v1 ~ Normal(0,1)
    v2 ~ Categorical(5)
end
```

# Use PG for a 'v2' variable, and use HMC for the 'v1' variable.
# Note that v2 is discrete, so the PG sampler is more appropriate
# than is HMC.
alg = Gibbs(HMC(0.2, 3, :v1), PG(20, :v2))
```

Tips:
- `HMC` and `NUTS` are fast samplers, and can throw off particle-based
methods like Particle Gibbs. You can increase the effectiveness of particle sampling by including
more particles in the particle sampler.
"""
struct Gibbs{space, A<:Tuple} <: InferenceAlgorithm
    algs::A   # component sampling algorithms

    function Gibbs{space, A}(algs::A) where {space, A<:Tuple}
        all(isgibbscomponent, algs) || error("all algorithms have to support Gibbs sampling")
        return new{space, A}(algs)
    end
end

function Gibbs(algs...)
    # obtain space of sampling algorithms
    space = Tuple(union(getspace.(algs)...))

    Gibbs{space, typeof(algs)}(algs)
end

"""
    GibbsState{V<:VarInfo, S<:Tuple{Vararg{Sampler}}}

Stores a `VarInfo` for use in sampling, and a `Tuple` of `Samplers` that
the `Gibbs` sampler iterates through for each `step!`.
"""
struct GibbsState{V<:VarInfo,S<:Tuple{Vararg{Sampler}},T}
    vi::V
    samplers::S
    states::T
end

struct GibbsTransition{T,F}
    "The parameters for any given sample."
    θ::T
    "The joint log probability for the sample's parameters."
    lp::F
end

function GibbsTransition(vi::AbstractVarInfo)
    theta = tonamedtuple(vi)
    lp = getlogp(vi)
    return GibbsTransition(theta, lp)
end

function additional_parameters(::Type{<:GibbsTransition})
    return [:lp]
end

DynamicPPL.getlogp(t::GibbsTransition) = t.lp

# extract varinfo object from state
getvarinfo(state) = state.vi
getvarinfo(state::AbstractVarInfo) = state

# update state with new varinfo object
gibbs_update_state(state::AbstractVarInfo, varinfo::AbstractVarInfo) = varinfo

# Initialize the Gibbs sampler.
function DynamicPPL.initialstep(
    rng::AbstractRNG,
    model::Model,
    spl::Sampler{<:Gibbs},
    vi::AbstractVarInfo;
    kwargs...
)
    # Create tuple of samplers
    i = 0
    algs = spl.alg.algs
    samplers = map(algs) do alg
        i += 1
        if i == 1
            prev_alg = algs[end]
        else
            prev_alg = algs[i-1]
        end
        rerun = !isa(alg, MH) || prev_alg isa PG || prev_alg isa ESS
        selector = Selector(Symbol(typeof(alg)), rerun)
        Sampler(alg, model, selector)
    end

    # Add Gibbs to gids for all variables.
    for sym in keys(vi.metadata)
        vns = getfield(vi.metadata, sym).vns

        for vn in vns
            # update the gid for the Gibbs sampler
            DynamicPPL.updategid!(vi, vn, spl)

            # try to store each subsampler's gid in the VarInfo
            for local_spl in samplers
                DynamicPPL.updategid!(vi, vn, local_spl)
            end
        end
    end

    # Compute initial states of the local samplers.
    states = map(samplers) do local_spl
        state = last(DynamicPPL.initialstep(rng, model, local_spl, vi; kwargs...))

        # update VarInfo object
        vi = getvarinfo(state)

        return state
    end

    # Compute initial transition and state.
    transition = GibbsTransition(vi)
    state = GibbsState(vi, samplers, states)

    return transition, state
end

# Subsequent steps
function AbstractMCMC.step(
    rng::AbstractRNG,
    model::Model,
    spl::Sampler{<:Gibbs},
    state::GibbsState;
    kwargs...
)
    @debug "Gibbs stepping..."

    # Iterate through each of the samplers.
    vi = state.vi
    samplers = state.samplers
    states = map(samplers, state.states) do _sampler, _state
        @debug "$(typeof(_sampler)) stepping..."

        # Update state of current sampler with updated `VarInfo` object.
        current_state = gibbs_update_state(_state, vi)

        # Step through the local sampler.
        newstate = last(AbstractMCMC.step(rng, model, _sampler, current_state; kwargs...))

        # Update `VarInfo` object.
        vi = getvarinfo(newstate)

        return newstate
    end

    return GibbsTransition(vi), GibbsState(vi, samplers, states)
end
