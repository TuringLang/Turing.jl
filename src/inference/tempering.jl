###########################
#   SAMPLING
###########################

"""
Upon calling sample on a `TemperedSampler`, we must manually instantiate a `Sampler`
to insert into the `TemperedSampler`, rather than wrapping the `TemperedSampler`
itself in a `Sampler`
"""
function create_tempered_sampler(
    model::AbstractModel,
    sampler::MCMCTempering.TemperedSampler{<:InferenceAlgorithm}
)
    return MCMCTempering.TemperedSampler(
        Sampler(sampler.internal_sampler, model),
        sampler.Δ,
        sampler.Δ_init,
        sampler.N_swap,
        sampler.swap_strategy
    )
end


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
    tempered_sampler = create_tempered_sampler(model, sampler)
    if resume_from === nothing
        return AbstractMCMC.sample(rng, model, tempered_sampler, N;
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
    tempered_sampler = create_tempered_sampler(model, sampler)
    return AbstractMCMC.sample(rng, model, tempered_sampler, ensemble, N, n_chains;
                               chain_type=chain_type, progress=progress, kwargs...)
end



###########################
#   MODEL
###########################

struct TemperedEval{T<:Real,F}
    f :: F
    β :: T
end

function (te::TemperedEval)(
    rng::Random.AbstractRNG,
    model::DynamicPPL.Model,
    varinfo::DynamicPPL.AbstractVarInfo,
    sampler::DynamicPPL.AbstractSampler,
    context::DynamicPPL.AbstractContext,
    args...
)
    minibatchcontext = DynamicPPL.MiniBatchContext(
        context,
        te.β
    )
    te.f(rng, model, varinfo, sampler, minibatchcontext, args...)
end

"""
    MCMCTempering.make_tempered_model(model, β)

Uses a `TemperedEval` to temper a `model`'s `f` by using a `DynamicPPL.MiniBatchContext`
to inject `β` as a multiplier on evaluation.
"""
function MCMCTempering.make_tempered_model(model::DynamicPPL.Model, β::Real)
    return DynamicPPL.Model{DynamicPPL.getmissings(model)}(
        model.name,
        TemperedEval(model.f, β),
        model.args,
        model.defaults
    )
end



###########################
#   SWAPPING
###########################

function MCMCTempering.get_tempered_loglikelihoods_and_params(
    model::DynamicPPL.Model,
    sampler::DynamicPPL.Sampler{<:InferenceAlgorithm},
    states,
    k::Integer,
    Δ::Vector{T},
    Δ_state::Vector{<:Integer}
) where {T<:AbstractFloat}

    logπk = MCMCTempering.make_tempered_loglikelihood(model, Δ[Δ_state[k]], sampler, get_vi(states[k][2]))
    logπkp1 = MCMCTempering.make_tempered_loglikelihood(model, Δ[Δ_state[k + 1]], sampler, get_vi(states[k + 1][2]))
    
    θk = MCMCTempering.get_params(states[k][2], sampler)
    θkp1 = MCMCTempering.get_params(states[k + 1][2], sampler)
    
    return logπk, logπkp1, θk, θkp1
end


"""
    MCMCTempering.make_tempered_loglikelihood(model, β, sampler, varinfo_init)

Construct the log likelihood function for a `model` weighted by inverse temperature `β`.
To do this we use the `sampler` to instantiate a new `VarInfo` alongside the current `varinfo_init`.
"""
function MCMCTempering.make_tempered_loglikelihood(
    model::DynamicPPL.Model,
    β::T,
    sampler::DynamicPPL.Sampler,
    varinfo_init::DynamicPPL.VarInfo
) where {T<:AbstractFloat}
    
    function logπ(z)
        varinfo = DynamicPPL.VarInfo(varinfo_init, sampler, z)
        model(varinfo)
        return DynamicPPL.getlogp(varinfo) * β
    end

    return logπ
end


"""
    get_vi(state) / get_vi(vi)

Returns the `VarInfo` given whatever a sampler's second return component is on an `AbstractMCMC.step`,
in some cases this is the `VarInfo` itself, in others it must be accessed as a property of the state.
"""
get_vi(state::Union{HMCState,GibbsState,EmceeState,SMCState}) = state.vi
get_vi(vi::DynamicPPL.VarInfo) = vi


"""
    MCMCTempering.get_params(state, sampler)

Uses the `sampler` to index the `VarInfo` extracted from the `state` and return the associated
`θ` parameter vector.
"""
MCMCTempering.get_params(state, sampler::DynamicPPL.Sampler) = get_vi(state)[sampler]
