###
### Sampler states
###

struct Emcee{space, E<:AMH.Ensemble} <: InferenceAlgorithm 
    ensemble::E
end

function Emcee(n_walkers::Int, stretch_length=2.0)
    # Note that the proposal distribution here is just a Normal(0,1)
    # because we do not need AdvancedMH to know the proposal for
    # ensemble sampling.
    prop = AMH.StretchProposal(nothing, stretch_length)
    ensemble = AMH.Ensemble(n_walkers, prop)
    return Emcee{(), typeof(ensemble)}(ensemble)
end

function Sampler(
    alg::Emcee,
    model::Model,
    s::Selector=Selector()
)
    # Set up info dict.
    info = Dict{Symbol, Any}()

    # Set up state struct.
    state = SamplerState(VarInfo(model))

    # Generate a sampler.
    return Sampler(alg, info, s, state)
end

alg_str(::Sampler{<:Emcee}) = "Emcee"

function AbstractMCMC.sample_init!(
    rng::AbstractRNG,
    model::Model,
    spl::Sampler{<:Emcee},
    N::Integer;
    verbose::Bool=true,
    resume_from=nothing,
    kwargs...
)
    # Resume the sampler.
    set_resume!(spl; resume_from=resume_from, kwargs...)

    # Get `init_theta`
    initialize_parameters!(spl; verbose=verbose, kwargs...)

    # Link everything before sampling.
    link!(spl.state.vi, spl)
end

function AbstractMCMC.sample_end!(
    rng::AbstractRNG,
    model::Model,
    spl::Sampler{<:Emcee},
    N::Integer,
    transitions;
    kwargs...
)
    # Invlink everything after sampling.
    invlink!(spl.state.vi, spl)
end

function _typed_draw(model, vi::VarInfo)
    spl = SampleFromPrior()
    empty!(vi)
    model(vi, spl)
    DynamicPPL.link!(vi, spl)
    params = vi[spl]
    return AMH.Transition(params, getlogp(vi))
end

function AbstractMCMC.step!(
    rng::AbstractRNG,
    model::Model,
    spl::Sampler{<:Emcee},
    N::Integer,
    transition::Nothing;
    kwargs...
)
    # Generate a log joint function.
    densitymodel = AMH.DensityModel(gen_logπ(spl.state.vi, DynamicPPL.SampleFromPrior(), model))

    # Make the first transition.
    walkers = map(x -> _typed_draw(model, spl.state.vi), 1:spl.alg.ensemble.n_walkers)
    
    return walkers
end

function AbstractMCMC.step!(
    rng::AbstractRNG,
    model::Model,
    spl::Sampler{<:Emcee},
    N::Integer,
    transition;
    kwargs...
)
    # Generate a log joint function.
    densitymodel = AMH.DensityModel(gen_logπ(spl.state.vi, DynamicPPL.SampleFromPrior(), model))

    # Make the first transition.
    new_transitions = AbstractMCMC.step!(rng, densitymodel, spl.alg.ensemble, 1, transition)
    return new_transitions
end

function transform_transition(spl::Sampler{<:Emcee}, ts::Vector{<:Vector}, w::Int, i::Int; linked=true)
    trans = ts[i][w]
    linked && DynamicPPL.link!(spl.state.vi, spl)
    spl.state.vi[spl] = trans.params
    linked && DynamicPPL.invlink!(spl.state.vi, spl)
    setlogp!(spl.state.vi, trans.lp)

    return Transition(spl)
end

function AbstractMCMC.bundle_samples(
    rng::AbstractRNG,
    model::AbstractModel,
    spl::Sampler{<:Emcee},
    N::Integer,
    ts::Vector{<:Vector},
    chain_type::Type{MCMCChains.Chains};
    save_state = false,
    kwargs...
)
    # Transform the transitions by linking them to the constrained space.
    ts_transform = [map(i -> transform_transition(spl, ts, w, i), 1:N) for w in 1:spl.alg.ensemble.n_walkers]

    # Construct individual chains by calling the default chain constructor.
    chains = map(
        w -> AbstractMCMC.bundle_samples(
                rng, 
                model, 
                spl, 
                N, 
                ts_transform[w], 
                chain_type; 
                save_state=save_state, 
                kwargs...),
        1:spl.alg.ensemble.n_walkers
    )

    # Concatenate all the chains.
    return reduce(MCMCChains.chainscat, chains)
end
