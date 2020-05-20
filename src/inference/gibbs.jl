###
### Gibbs samplers / compositional samplers.
###

const GibbsComponent = Union{Hamiltonian,MH,ESS,PG}

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
struct Gibbs{space, A<:Tuple{Vararg{GibbsComponent}}} <: InferenceAlgorithm
    algs::A   # component sampling algorithms
end

function Gibbs(algs::GibbsComponent...)
    # obtain space of sampling algorithms
    space = Tuple(union(getspace.(algs)...))

    Gibbs{space, typeof(algs)}(algs)
end

"""
    GibbsState{V<:AbstractVarInfo, S<:Tuple{Vararg{Sampler}}}

Stores a `VarInfo` for use in sampling, and a `Tuple` of `Samplers` that
the `Gibbs` sampler iterates through for each `step!`.
"""
mutable struct GibbsState{V<:AbstractVarInfo, S<:Tuple{Vararg{Sampler}}} <: AbstractSamplerState
    vi::V
    samplers::S
end

function GibbsState(model::Model, samplers::Tuple{Vararg{Sampler}})
    return GibbsState(VarInfo(model), samplers)
end

function replace_varinfo(s::GibbsState, vi::AbstractVarInfo)
    return GibbsState(vi, s.samplers)
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
    varinfo = merge(ntuple(i -> samplers[i].state.vi, Val(length(samplers)))...)
    samplers = map(samplers) do sampler
        Sampler(
            sampler.alg,
            sampler.info,
            sampler.selector,
            replace_varinfo(sampler.state, varinfo),
        )
    end
    # create a state variable
    state = GibbsState(varinfo, samplers)

    # create the sampler
    info = Dict{Symbol, Any}()
    spl = Sampler(alg, info, s, state)

    # add Gibbs to gids for all variables
    DynamicPPL.updategid!(varinfo, (spl, samplers...))

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

function Base.promote_type(
    ::Type{GibbsTransition{T1, F1, S1}},
    ::Type{GibbsTransition{T2, F2, S2}},
) where {T1, F1, S1, T2, F2, S2}
    return GibbsTransition{
        Union{T1, T2},
        promote_type(F1, F2),
        promote_type(S1, S2),
    }
end
function Base.convert(
    ::Type{GibbsTransition{T, F, S}},
    t::GibbsTransition,
) where {T, F, S}
    return GibbsTransition{T, F, S}(
        convert(T, t.θ),
        convert(F, t.lp),
        convert(S, t.transitions),
    )
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
    Turing.DEBUG && @debug "Gibbs stepping..."

    # Iterate through each of the samplers.
    transitions = map(enumerate(spl.state.samplers)) do (i, local_spl)
        Turing.DEBUG && @debug "$(typeof(local_spl)) stepping..."

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
function AbstractMCMC.transitions(
    transition::GibbsTransition,
    ::Model,
    ::Sampler{<:Gibbs},
    N::Integer;
    kwargs...
)
    ts = Vector{Transition{typeof(transition.θ),typeof(transition.lp)}}(undef, 0)
    sizehint!(ts, N)
    return ts
end

function AbstractMCMC.save!!(
    transitions::Vector{<:Transition},
    transition::GibbsTransition,
    iteration::Integer,
    ::Model,
    ::Sampler{<:Gibbs},
    ::Integer;
    kwargs...
)
    return BangBang.push!!(transitions, Transition(transition.θ, transition.lp))
end
