###########################
#   SAMPLING
###########################

"""
Upon calling sample on a `TemperedSampler`, we must manually instantiate a `Sampler`
to insert into the `TemperedSampler`, rather than wrapping the `TemperedSampler`
itself in a `Sampler`
"""
function AbstractMCMC.sample(
    rng::AbstractRNG,
    model::AbstractModel,
    sampler::MCMCTempering.TemperedSampler{<:InferenceAlgorithm},
    N::Integer;
    chain_type=MCMCChains.Chains,
    resume_from=nothing,
    progress=PROGRESS[],
    kwargs...
)
    sampler = MCMCTempering.TemperedSampler(
        Sampler(sampler.internal_sampler, model),
        sampler.Δ,
        sampler.Δ_init,
        sampler.N_swap,
        sampler.swap_strategy
    )

    if resume_from === nothing
        return AbstractMCMC.mcmcsample(rng, model, sampler, N;
                                       chain_type=chain_type, progress=progress, kwargs...)
    else
        return resume(resume_from, N; chain_type=chain_type, progress=progress, kwargs...)
    end
end


function AbstractMCMC.sample(
    rng::AbstractRNG,
    model::AbstractModel,
    sampler::MCMCTempering.TemperedSampler{<:InferenceAlgorithm},
    ensemble::AbstractMCMC.AbstractMCMCEnsemble,
    N::Integer,
    n_chains::Integer;
    chain_type=MCMCChains.Chains,
    progress=PROGRESS[],
    kwargs...
)
    sampler = MCMCTempering.TemperedSampler(
        Sampler(sampler.internal_sampler, model),
        sampler.Δ,
        sampler.Δ_init,
        sampler.N_swap,
        sampler.swap_strategy
    )

    return AbstractMCMC.mcmcsample(rng, model, sampler, ensemble, N, n_chains;
                                   chain_type=chain_type, progress=progress, kwargs...)
end


function AbstractMCMC.bundle_samples(
    ts::Vector,
    model::DynamicPPL.Model,
    spl::MCMCTempering.TemperedSampler,
    state::MCMCTempering.TemperedState,
    chain_type::Union{Type{MCMCChains.Chains},Type{Vector{NamedTuple}}};
    kwargs...
)
    return AbstractMCMC.bundle_samples(ts, model, spl.internal_sampler, state, chain_type; kwargs...)
end



###########################
#   MODEL
###########################

struct TemperedEval{T<:AbstractFloat}
    model :: DynamicPPL.Model
    β     :: T
end

function (te::TemperedEval)(
    rng::Random.AbstractRNG,
    model::DynamicPPL.Model,
    varinfo::DynamicPPL.AbstractVarInfo,
    sampler::DynamicPPL.AbstractSampler,
    context::DynamicPPL.AbstractContext,
    args...
)
    context = DynamicPPL.MiniBatchContext(
        context,
        te.β
    )
    te.model.f(rng, model, varinfo, sampler, context, args...)
end

"""
    MCMCTempering.make_tempered_model

Uses a `TemperedEval` to temper a `model`'s `f` by using a `DynamicPPL.MiniBatchContext` to inject `β` as a multiplier on evaluation
"""
function MCMCTempering.make_tempered_model(model::DynamicPPL.Model, β::T) where {T<:AbstractFloat}
    return DynamicPPL.Model(model.name, TemperedEval(model, β), model.args, model.defaults)
end



###########################
#   SWAPPING
###########################

function MCMCTempering.get_densities_and_θs(
    model::DynamicPPL.Model,
    sampler::DynamicPPL.Sampler{<:InferenceAlgorithm},
    states,
    k::Integer,
    Δ::Vector{T},
    Δ_state::Vector{<:Integer}
) where {T<:AbstractFloat}

    logπk = MCMCTempering.make_tempered_logπ(model, Δ[Δ_state[k]], sampler, get_vi(states[k][2]))
    logπkp1 = MCMCTempering.make_tempered_logπ(model, Δ[Δ_state[k + 1]], sampler, get_vi(states[k + 1][2]))
    
    θk = MCMCTempering.get_θ(states[k][2], sampler)
    θkp1 = MCMCTempering.get_θ(states[k + 1][2], sampler)
    
    return logπk, logπkp1, θk, θkp1
end


"""
    MCMCTempering.make_tempered_logπ

Constructs the likelihood density function for a `model` weighted by inverse temperature `β`

# Arguments
- The `model` in question
- An inverse temperature `β` with which to weight the density
- A `sampler` to instantiate a new `VarInfo`
- Alongside the current `varinfo_init`

## Notes
- For sake of efficiency, the returned function is closed over an instance of `VarInfo`. This means that you *might* run into some weird behaviour if you call this method sequentially using different types; if that's the case, just generate a new one for each type using `make_`.
"""
function MCMCTempering.make_tempered_logπ(model::DynamicPPL.Model, β::T, sampler::DynamicPPL.Sampler, varinfo_init::DynamicPPL.VarInfo) where {T<:AbstractFloat}
    
    function logπ(z)
        varinfo = DynamicPPL.VarInfo(varinfo_init, sampler, z)
        model(varinfo)
        return DynamicPPL.getlogp(varinfo) * β
    end

    return logπ
end


"""
    get_vi

Returns the `VarInfo` given whatever a sampler's second return component is on an `AbstractMCMC.step`, in some cases this is the `VarInfo` itself, in others it must be accessed as a property of the state
"""
get_vi(state::Union{HMCState,GibbsState,EmceeState,SMCState}) = state.vi
get_vi(vi::DynamicPPL.VarInfo) = vi


"""
MCMCTempering.get_θ

Uses the `sampler` to index the `VarInfo` and return the associated `θ` proposal
"""
MCMCTempering.get_θ(state, sampler::DynamicPPL.Sampler) = get_vi(state)[sampler]
# get_vi(states, k)[DynamicPPL.SampleFromPrior()]
