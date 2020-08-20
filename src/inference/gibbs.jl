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
mutable struct GibbsState{V<:VarInfo, S<:Tuple{Vararg{Sampler}}} <: AbstractSamplerState
    vi::V
    samplers::S
end

function GibbsState(model::Model, samplers::Tuple{Vararg{Sampler}})
    return GibbsState(VarInfo(model), samplers)
end

function Sampler(alg::Gibbs, model::Model, s::Selector)
    # sanity check for space
    space = getspace(alg)
    # create tuple of samplers
    i = 0
    samplers = map(alg.algs) do _alg
        i += 1
        if i == 1
            prev_alg = alg.algs[end]
        else
            prev_alg = alg.algs[i-1]
        end
        rerun = !(_alg isa MH) || prev_alg isa PG || prev_alg isa ESS
        selector = Selector(Symbol(typeof(_alg)), rerun)
        Sampler(_alg, model, selector)
    end
    # create a state variable
    state = GibbsState(model, samplers)

    # create the sampler
    info = Dict{Symbol, Any}()
    spl = Sampler(alg, info, s, state)

    # add Gibbs to gids for all variables
    vi = spl.state.vi
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

    return spl
end

"""
    GibbsTransition

Fields:
- `θ`: The parameters for any given sample.
- `lp`: The log pdf for the sample's parameters.
- `transitions`: The transitions of the samplers.
"""
struct GibbsTransition{T,F,S<:AbstractVector}
    θ::T
    lp::F
    transitions::S
end

function GibbsTransition(spl::Sampler{<:Gibbs}, transitions::AbstractVector)
    theta = tonamedtuple(spl.state.vi)
    lp = getlogp(spl.state.vi)
    return GibbsTransition(theta, lp, transitions)
end

function additional_parameters(::Type{<:GibbsTransition})
    return [:lp]
end

DynamicPPL.getlogp(t::GibbsTransition) = t.lp

# Initialize the Gibbs sampler.
function AbstractMCMC.sample_init!(
    rng::AbstractRNG,
    model::Model,
    spl::Sampler{<:Gibbs},
    N::Integer;
    kwargs...
)
    # Initialize each local sampler.
    for local_spl in spl.state.samplers
        AbstractMCMC.sample_init!(rng, model, local_spl, N; kwargs...)
    end
end

# Finalize the Gibbs sampler.
function AbstractMCMC.sample_end!(
    rng::AbstractRNG,
    model::Model,
    spl::Sampler{<:Gibbs},
    N::Integer;
    kwargs...
)
    # Finalize each local sampler.
    for local_spl in spl.state.samplers
        AbstractMCMC.sample_end!(rng, model, local_spl, N; kwargs...)
    end
end

# Steps 2
function AbstractMCMC.step!(
    rng::AbstractRNG,
    model::Model,
    spl::Sampler{<:Gibbs},
    N::Integer,
    transition::Union{Nothing,GibbsTransition};
    kwargs...
)
    @debug "Gibbs stepping..."

    # Iterate through each of the samplers.
    transitions = map(enumerate(spl.state.samplers)) do (i, local_spl)
        @debug "$(typeof(local_spl)) stepping..."

        # Update the sampler's VarInfo.
        local_spl.state.vi = spl.state.vi

        # Step through the local sampler.
        if transition === nothing
            trans = AbstractMCMC.step!(rng, model, local_spl, N, nothing; kwargs...)
        else
            trans = AbstractMCMC.step!(rng, model, local_spl, N, transition.transitions[i];
                                       kwargs...)
        end

        # After the step, update the master varinfo.
        spl.state.vi = local_spl.state.vi

        trans
    end

    return GibbsTransition(spl, transitions)
end

# Do not store transitions of subsamplers
function AbstractMCMC.transitions_init(
    transition::GibbsTransition,
    ::Model,
    ::Sampler{<:Gibbs},
    N::Integer;
    kwargs...
)
    return Vector{Transition{typeof(transition.θ),typeof(transition.lp)}}(undef, N)
end

function AbstractMCMC.transitions_save!(
    transitions::Vector{<:Transition},
    iteration::Integer,
    transition::GibbsTransition,
    ::Model,
    ::Sampler{<:Gibbs},
    ::Integer;
    kwargs...
)
    transitions[iteration] = Transition(transition.θ, transition.lp)
    return
end
