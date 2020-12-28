###
### Gibbs samplers / compositional samplers.
###


"""
    isgibbscomponent(alg)

Determine whether algorithm `alg` is allowed as a Gibbs component.
"""
isgibbscomponent(alg) = false

isgibbscomponent(::ESS) = true
isgibbscomponent(::GibbsConditional) = true
isgibbscomponent(::Hamiltonian) = true
isgibbscomponent(::MH) = true
isgibbscomponent(::PG) = true

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

metadata(t::GibbsTransition) = (lp = t.lp,)

DynamicPPL.getlogp(t::GibbsTransition) = t.lp

# extract varinfo object from state
"""
    gibbs_varinfo(model, sampler, state)

Return the variables corresponding to the current `state` of the Gibbs component `sampler`.
"""
gibbs_varinfo(model, sampler, state) = varinfo(state)
varinfo(state) = state.vi
varinfo(state::AbstractVarInfo) = state

"""
    gibbs_state(model, sampler, state, varinfo)

Return the state of the Gibbs component with updated variables `varinfo`.

The Gibbs component samples from the `model` using the provided `sampler` and its current
state is `state`.
"""
gibbs_state(model, sampler, state::AbstractVarInfo, varinfo::AbstractVarInfo) = varinfo

# Update state in Gibbs sampling
function gibbs_state(
    model::Model,
    spl::Sampler{<:Hamiltonian},
    state::HMCState,
    varinfo::AbstractVarInfo,
)
    # Update hamiltonian
    θ_old = varinfo[spl]
    hamiltonian = get_hamiltonian(model, spl, varinfo, state, length(θ_old))

    # TODO: Avoid mutation
    resize!(state.z.θ, length(θ_old))
    state.z.θ .= θ_old
    z = state.z

    return HMCState(varinfo, state.i, state.traj, hamiltonian, z, state.adaptor)
end

"""
    gibbs_rerun(prev_sampler, sampler)

Check if the model should be rerun to recompute the log density before sampling with the
Gibbs component `sampler` and after sampling from Gibbs component `prev_sampler`.

By default, the function returns `true`.
"""
gibbs_rerun(prev_sampler, sampler) = true

# `vi.logp` already contains the log joint probability if the previous sampler
# used a `GibbsConditional` or one of the standard `Hamiltonian` algorithms
gibbs_rerun(::Sampler{<:GibbsConditional}, ::Sampler{<:MH}) = false
gibbs_rerun(::Sampler{<:Hamiltonian}, ::Sampler{<:MH}) = false

# `vi.logp` already contains the log joint probability if the previous sampler
# used a `GibbsConditional` or a `MH` algorithm
gibbs_rerun(::Sampler{<:MH}, ::Sampler{<:Hamiltonian}) = false
gibbs_rerun(::Sampler{<:GibbsConditional}, ::Sampler{<:Hamiltonian}) = false

# do not have to recompute `vi.logp` since it is not used in `step`
gibbs_rerun(prev_sampler, ::Sampler{<:GibbsConditional}) = false

# Do not recompute `vi.logp` since it is reset anyway in `step`
gibbs_rerun(prev_sampler, ::Sampler{<:PG}) = false

# Initialize the Gibbs sampler.
function DynamicPPL.initialstep(
    rng::AbstractRNG,
    model::Model,
    spl::Sampler{<:Gibbs},
    vi::AbstractVarInfo;
    kwargs...
)
    # Create tuple of samplers
    algs = spl.alg.algs
    samplers = map(algs) do alg
        Sampler(alg, model)
    end

    # Compute initial states of the local samplers.
    i = 0
    states = map(samplers) do local_spl
        # Recompute `vi.logp` if needed.
        prev_sampler = (i += 1) == 1 ? samplers[end] : samplers[i-1]
        if gibbs_rerun(prev_sampler, local_spl)
            model(rng, vi, local_spl)
        end

        # Compute initial state.
        _, state = DynamicPPL.initialstep(rng, model, local_spl, vi; kwargs...)

        # Update `VarInfo` object.
        vi = gibbs_varinfo(model, local_spl, state)

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
    # Iterate through each of the samplers.
    vi = state.vi
    samplers = state.samplers
    i = 0
    states = map(samplers, state.states) do _sampler, _state
        # Recompute `vi.logp` if needed.
        prev_sampler = (i += 1) == 1 ? samplers[end] : samplers[i-1]
        if gibbs_rerun(prev_sampler, _sampler)
            model(rng, vi, _sampler)
        end

        # Update state of current sampler with updated `VarInfo` object.
        current_state = gibbs_state(model, _sampler, _state, vi)

        # Step through the local sampler.
        _, newstate = AbstractMCMC.step(rng, model, _sampler, current_state; kwargs...)

        # Update `VarInfo` object.
        vi = gibbs_varinfo(model, _sampler, newstate)

        return newstate
    end

    return GibbsTransition(vi), GibbsState(vi, samplers, states)
end
