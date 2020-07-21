using LinearAlgebra

struct GetData <: InferenceAlgorithm end

struct DataState{T} <: DynamicPPL.AbstractSamplerState
    observations::T
end

DynamicPPL.getspace(::Sampler{<:GetData}) = ()

function DynamicPPL.assume(rng, spl::Sampler{<:GetData}, dist::Distribution, vn::DynamicPPL.VarName, vi)
    r = rand(rng, dist)
    push!(vi, vn, r, dist, spl)
    return r, 0
end

function DynamicPPL.observe(spl::Sampler{<:GetData}, dist::Distribution, value, vi)
    push!(spl.state.observations, value)
    return 0
end

function DataState(model::DynamicPPL.Model)
    return DataState([])
end

DynamicPPL.Sampler(alg::GetData, model::Model, s::DynamicPPL.Selector) = Sampler(alg, Dict{Symbol, Any}(), s, DataState(model))

function get_data(model::Model)
    # Get the data by executing the model once
    spl = Sampler(GetData(), model, DynamicPPL.Selector())
    var_info = Turing.VarInfo()
    model(var_info, spl)

    return [x for x in spl.state.observations] # trying to infer type
end

###################
### AbstractABC ###
###################
abstract type AbstractABC <: InferenceAlgorithm end
DynamicPPL.getspace(::Sampler{<:AbstractABC}) = ()

function DynamicPPL.assume(rng, spl::Sampler{<:AbstractABC}, dist::Distribution, vn::VarName, vi)
    # Sample from prior
    r = rand(rng, dist)
    push!(vi, vn, r, dist, spl)
    return r, 0
end

function DynamicPPL.observe(spl::Sampler{<:AbstractABC}, dist::Distribution, value, vi)
    # 1. Sample from likelihood
    proposed = rand(dist)

    # HACK: 2. Store proposal in sampler state
    push!(spl.state.proposed, proposed)

    return 0
end

function AbstractMCMC.sample_end!(
    ::AbstractRNG,
    ::Model,
    spl::Sampler{<:AbstractABC},
    N::Integer,
    ts::Vector{T};
    kwargs...
) where {T}
    # HACK: rejecting samples after full sampling
    results = T[]
    push!(results, ts[1])

    num_accepted = 1

    for i = 2:length(ts)
        if ts[i] !== ts[i - 1]
            # if they're different, we accepted
            push!(results, ts[i])
            num_accepted += 1
        # else
        #     @info "Rejecting $(i) because $(ts[i].θ) === $(ts[i - 1].θ)"
        end
    end

    if num_accepted == 0
        error("No samples were accepted; try increasing the acceptance threshold.")
    elseif num_accepted < length(ts)
        @warn "Only accepted $(100 * num_accepted / length(ts))% of the samples"
    end

    empty!(ts)
    append!(ts, results)
end


###########
### ABC ###
###########
struct ABC{A, F1, F2, T} <: AbstractABC
    proposal::A
    distance::F1
    stat::F2
    epsilon::T
end
function ABC(proposal, dist; stat = identity, epsilon = 0.1)
    return ABC(proposal, dist, stat, epsilon)
end

struct ABCState{V<:VarInfo, A1, A2} <: AbstractSamplerState
    vi::V
    observations::A1
    proposed::A2
end

function ABCState(model::Model)
    data = get_data(model)
    proposals = similar(data)
    empty!(proposals)

    return ABCState(VarInfo(model), data, proposals)
end

function Sampler(alg::ABC, model::Model, s::Selector)
    info = Dict{Symbol, Any}()
    state = ABCState(model)
    return Sampler(alg, info, s, state)
end

function AbstractMCMC.step!(
    rng::AbstractRNG,
    model::Model,
    spl::Sampler{<:ABC},
    ::Integer,
    transition;
    kwargs...
)
    # TODO: add some limit
    num_iters = 0
    while true
        # Sample from model
        empty!(spl.state.proposed)
        empty!(spl.state.vi)

        alg = spl.alg
        state = spl.state

        # 1. Propose parameters
        # HACK: dunno
        densitymodel = AMH.DensityModel(identity)
        # Propose
        θ_new = AMH._propose(rng, alg.proposal, densitymodel)
        # HACK: Update
        ranges = DynamicPPL._getranges(spl.state.vi, DynamicPPL.Selector(), Val(DynamicPPL.getspace(spl)))
        num_params = length(spl.state.vi[spl])
        tmp = zeros(num_params)
        # TODO: `@generated`
        for k in keys(θ_new)
            tmp[ranges[k]] .= θ_new[k]
        end
        spl.state.vi[spl] = tmp

        # 2. Run model to get proposals
        model(rng, spl.state.vi, spl)

        # 3. Accept/reject
        if isnothing(transition)
            # Early return if this is the first sample
            return Transition(spl)
        else
            # If distance is within the required threshold, we update
            if alg.distance(alg.stat(state.proposed), alg.stat(state.observations)) ≤ alg.epsilon
                return Transition(spl)
            # else
            #     # Return old sample
            #     return transition
            end
        end

        num_iters += 1
        if num_iters == 10
            @warn "[ABC] ≥10 samples rejected before acceptance; maybe increase acceptance threshold?"
        end
    end
end
