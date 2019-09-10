###
### Gibbs samplers / compositional samplers.
###

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
alg = Gibbs(1000, HMC(1, 0.2, 3, :v1), PG(20, 1, :v2))
```

Tips:
- `HMC` and `NUTS` are fast samplers, and can throw off particle-based
methods like Particle Gibbs. You can increase the effectiveness of particle sampling by including
more particles in the particle sampler.
"""
mutable struct Gibbs{A} <: InferenceAlgorithm
    algs      ::  A   # component sampling algorithms
    function Gibbs(algs...)
        return new{typeof(algs)}(algs)
    end
end

alg_str(::Sampler{<:Gibbs}) = "Gibbs"
transition_type(spl::Sampler{<:Gibbs}) = typeof(Transition(spl))

"""
    GibbsState{V<:VarInfo, S<:Tuple{Vararg{Sampler}}}

Stores a `VarInfo` for use in sampling, and a `Tuple` of `Samplers` that
the `Gibbs` sampler iterates through for each `step!`.
"""
mutable struct GibbsState{V<:VarInfo, S<:Tuple{Vararg{Sampler}}} <: AbstractSamplerState
    vi::V
    samplers::S
end

function GibbsState(model::Model, samplers::S) where S<:Tuple{Vararg{Sampler}}
    return GibbsState(VarInfo(model), samplers)
end

const GibbsComponent = Union{Hamiltonian,MH,PG}

function Sampler(alg::Gibbs, model::Model, s::Selector)
    info = Dict{Symbol, Any}()

    n_samplers = length(alg.algs)
    samplers = Array{Sampler}(undef, n_samplers)
    space = Set{Symbol}()

    for i in 1:n_samplers
        sub_alg = alg.algs[i]
        if isa(sub_alg, GibbsComponent)
            samplers[i] = Sampler(sub_alg, model, Selector(Symbol(typeof(sub_alg))))
        else
            @error("[Gibbs] Unsupported sampling algorithm $sub_alg")
        end
        space = (space..., getspace(sub_alg)...)
    end

    # Sanity check for space
    @assert issubset(get_pvars(model), space) "[Gibbs] symbols specified to samplers ($space) doesn't cover the model parameters ($(get_pvars(model)))"

    if !(issetequal(get_pvars(model), space))
        @warn("[Gibbs] extra parameters specified by samplers don't exist in model: $(setdiff(space, get_pvars(model)))")
    end

    # Create a state variable.
    state = GibbsState(model, tuple(samplers...))

    # Create the sampler.
    spl = Sampler(alg, info, s, state)

    # Add Gibbs to gids for all variables.
    for sym in keys(spl.state.vi.metadata)
        vns = getfield(spl.state.vi.metadata, sym).vns
        for vn in vns
            # Update the gid for the Gibbs sampler.
            Turing.RandomVariables.updategid!(spl.state.vi, vn, spl)
            
            # Try to store each subsampler's gid in the VarInfo.
            for local_spl in spl.state.samplers
                Turing.RandomVariables.updategid!(spl.state.vi, vn, local_spl)
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
    lp = nothing; ϵ = nothing; eval_num = nothing

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
    t::TransitionType;
    kwargs...
) where TransitionType<:AbstractTransition
    Turing.DEBUG && @debug "Gibbs stepping..."

    time_elapsed = 0.0
    lp = nothing 
    ϵ = nothing
    eval_num = nothing

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
