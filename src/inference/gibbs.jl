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

# Initialize the Gibbs sampler.
function sample_init!(
    rng::AbstractRNG,
    model::Model,
    spl::Sampler{<:Gibbs},
    N::Integer;
    kwargs...
)
    # Initialize each local sampler.
    for local_spl in spl.state.samplers
        sample_init!(rng, model, local_spl, N; kwargs...)
    end
end

# Finalize the Gibbs sampler.
function sample_end!(
    rng::AbstractRNG,
    model::Model,
    spl::Sampler{<:Gibbs},
    N::Integer;
    kwargs...
)
    # Finalize each local sampler.
    for local_spl in spl.state.samplers
        sample_end!(rng, model, local_spl, N; kwargs...)
    end
end


# First step.
function step!(
    rng::AbstractRNG,
    model::Model,
    spl::Sampler{<:Gibbs},
    N::Integer;
    kwargs...
)
    Turing.DEBUG && @debug "Gibbs stepping..."

    time_elapsed = 0.0

    # Iterate through each of the samplers.
    for local_spl in spl.state.samplers
        Turing.DEBUG && @debug "$(typeof(local_spl)) stepping..."

        Turing.DEBUG && @debug "recording old θ..."

        # Update the sampler's VarInfo.
        local_spl.state.vi = spl.state.vi

        # Step through the local sampler.
        time_elapsed_thin =
            @elapsed step!(rng, model, local_spl, N; kwargs...)

        # After the step, update the master varinfo.
        spl.state.vi = local_spl.state.vi

        # Uncomment when developing thinning functionality.
        # Retrieve symbol to store this subsample.
        # symbol_id = Symbol(local_spl.selector.gid)

        # # Store the subsample.
        # spl.state.subsamples[symbol_id][] = trans

        # Record elapsed time.
        time_elapsed += time_elapsed_thin
    end

    return Transition(spl)
end

# Steps 2:N
function step!(
    rng::AbstractRNG,
    model::Model,
    spl::Sampler{<:Gibbs},
    N::Integer,
    t::AbstractTransition;
    kwargs...
)
    Turing.DEBUG && @debug "Gibbs stepping..."

    time_elapsed = 0.0

    # Iterate through each of the samplers.
    for local_spl in spl.state.samplers
        Turing.DEBUG && @debug "$(typeof(local_spl)) stepping..."

        Turing.DEBUG && @debug "recording old θ..."

        # Update the sampler's VarInfo.
        local_spl.state.vi = spl.state.vi

        # Step through the local sampler.
        time_elapsed_thin =
            @elapsed trans = step!(rng, model, local_spl, N, t; kwargs...)

        # After the step, update the master varinfo.
        spl.state.vi = local_spl.state.vi

        # Uncomment when developing thinning functionality.
        # Retrieve symbol to store this subsample.
        # symbol_id = Symbol(local_spl.selector.gid)
        #
        # # Store the subsample.
        # spl.state.subsamples[symbol_id][] = trans

        # Record elapsed time.
        time_elapsed += time_elapsed_thin
    end

    return Transition(spl)
end
