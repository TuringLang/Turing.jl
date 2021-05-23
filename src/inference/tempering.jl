###########################
#   SAMPLER
###########################

function AbstractMCMC.sample(
    rng::AbstractRNG,
    model::AbstractModel,
    sampler::TemperedSampler{<:InferenceAlgorithm},
    N::Integer;
    chain_type=MCMCChains.Chains,
    resume_from=nothing,
    progress=PROGRESS[],
    kwargs...
)
    sampler = TemperedSampler(Sampler(sampler, model), sampler.Δ, sampler.Δ_init, sampler.N_swap, sampler.swap_strategy)

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
    sampler::TemperedSampler{<:InferenceAlgorithm},
    ensemble::AbstractMCMC.AbstractMCMCEnsemble,
    N::Integer,
    n_chains::Integer;
    chain_type=MCMCChains.Chains,
    progress=PROGRESS[],
    kwargs...
)
    sampler = TemperedSampler(Sampler(sampler, model), sampler.Δ, sampler.Δ_init, sampler.N_swap, sampler.swap_strategy)

    return AbstractMCMC.mcmcsample(rng, model, sampler, ensemble, N, n_chains;
                                   chain_type=chain_type, progress=progress, kwargs...)
end


function AbstractMCMC.bundle_samples(
    ts::Vector,
    model::DynamicPPL.Model,
    spl::TemperedSampler,
    state::TemperedState,
    chain_type::Union{Type{MCMCChains.Chains},Type{Vector{NamedTuple}}};
    kwargs...
)
    return AbstractMCMC.bundle_samples(ts, model, spl.internal_sampler, state, chain_type; kwargs...)
end



###########################
#   MODEL
###########################

struct TemperedEval
    model :: DynamicPPL.Model
    β     :: AbstractFloat
end

function (te::TemperedEval)(
    rng,
    model,
    varinfo,
    sampler,
    context,
    args...
)
    context = DynamicPPL.MiniBatchContext(
        context,
        te.β
    )
    te.model.f(rng, model, varinfo, sampler, context, args...)
end


function MCMCTempering.make_tempered_model(model::Model, β::T) where {T<:AbstractFloat}
    return DynamicPPL.Model(model.name, TemperedEval(model, β), model.args, model.defaults)
end



###########################
#   SWAPPING
###########################

# function get_sampler(sampler::DynamicPPL.Sampler{<:MCMCTempering.TemperedSampler})
#     return DynamicPPL.Sampler(sampler.alg.alg, model, sampler.selector)
# end

function MCMCTempering.get_densities_and_θs(
    model::Model,
    sampler::Sampler{<:TemperedAlgorithm},
    states,
    k::Integer,
    Δ::Vector{T},
    Δ_state::Vector{<:Integer}
) where {T<:AbstractFloat}

    logπk = make_tempered_logπ(model, Δ[Δ_state[k]], sampler, get_vi(states[k][2]))
    logπkp1 = make_tempered_logπ(model, Δ[Δ_state[k + 1]], sampler, get_vi(states[k + 1][2]))
    
    θk = get_θ(states[k][2], sampler)
    θkp1 = get_θ(states[k + 1][2], sampler)
    
    return logπk, logπkp1, θk, θkp1
end


"""
    make_tempered_logπ

Constructs the likelihood density function for a `model` weighted by `β`

# Arguments
- The `model` in question
- An inverse temperature `β` with which to weight the density

## Notes
- For sake of efficiency, the returned function is closed over an instance of `VarInfo`. This means that you *might* run into some weird behaviour if you call this method sequentially using different types; if that's the case, just generate a new one for each type using `make_`.
"""
function MCMCTempering.make_tempered_logπ(model::Model, β::T, sampler::DynamicPPL.Sampler, varinfo_init::DynamicPPL.VarInfo) where {T<:AbstractFloat}
    
    function logπ(z)
        varinfo = DynamicPPL.VarInfo(varinfo_init, sampler, z)
        model(varinfo)
        return DynamicPPL.getlogp(varinfo) * β
    end

    return logπ
end


"""
    get_vi

Returns the `VarInfo` portion of the `k`th chain's state contained in `states`

# Arguments
- `states` is 2D array containing `length(Δ)` pairs of transition + state for each chain
- `k` is the index of a chain in `states`
"""
get_vi(state::Union{HMCState,GibbsState,EmceeState,SMCState}) = state.vi
get_vi(vi::DynamicPPL.VarInfo) = vi


"""
    get_θ

Uses the `sampler` to index the `VarInfo` of the `k`th chain and return the associated `θ` proposal

# Arguments
- `states` is 2D array containing `length(Δ)` pairs of transition + state for each chain
- `k` is the index of a chain in `states`
- `sampler` is used to index the `VarInfo` such that `θ` is returned
"""
MCMCTempering.get_θ(state, sampler::DynamicPPL.Sampler) = get_vi(state)[sampler]
# get_vi(states, k)[DynamicPPL.SampleFromPrior()]
