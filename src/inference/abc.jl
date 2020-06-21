using LinearAlgebra
using GpABC

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

    # HACK: 2. Store proposal and observation (`value`) in sampler state so that we can later compute distance
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
        # @info "stuff" length(spl.state.proposed) length(spl.state.observations) alg.distance(alg.stat(state.proposed), alg.stat(state.observations)) mean(state.proposed) mean(state.observations) transition.θ.m[1]
        if alg.distance(alg.stat(state.proposed), alg.stat(state.observations)) > alg.epsilon
            # Return old sample
            return transition
        else
            return Transition(spl)
        end
    end
end


#############
### GPABC ###
#############
struct GPABC{F1, F2, T} <: AbstractABC
    summary_statistic::F1
    distance_function::F2
    threshold::T
    n_particles::Int64
    max_iter::Int64
end

struct GPABCState{V, A1, A2} <: AbstractSamplerState
    vi::V
    proposed::A1
    reference_summary_statistic::A2
end

function GPABCState(alg::Turing.Inference.GPABC, model::Model)
    vi = Turing.VarInfo(model)

    observations = get_data(model)
    proposed = similar(observations)
    empty!(proposed)

    reference_summary_statistic = alg.summary_statistic(reshape(observations, (:, 1)))

    return GPABCState(vi, proposed, reference_summary_statistic)
end

Sampler(alg::GPABC, model::Model, s::DynamicPPL.Selector) = Sampler(alg, Dict{Symbol, Any}(), s, GPABCState(alg, model))

function AbstractMCMC.step!(
    rng::AbstractRNG,
    model::Model,
    spl::Sampler{<:GPABC},
    ::Integer,
    transition;
    kwargs...
)
    empty!(spl.state.proposed)
    empty!(spl.state.vi)

    state = spl.state
    alg = spl.alg

    θ = state.vi[spl]
    logweight = getlogp(state.vi)

    function simulator_function(θ)
        # Sample from model
        empty!(spl.state.proposed)
        empty!(spl.state.vi)

        spl.state.vi[spl] .= θ
        model(spl.state.vi, spl)

        # FIXME: current impl isn't going to work in general
        if (spl.state.proposed isa AbstractVector{<:Real})
            return reshape(deepcopy(spl.state.proposed), (:, 1))
        else
            return deepcopy(spl.state.proposed)
        end
    end

    distance = GpABC.simulate_distance(
        reshape(θ, (1, :)),
        simulator_function,
        alg.summary_statistic,
        state.reference_summary_statistic,
        alg.distance_function
    )

    if isnothing(transition)
        return Transition(spl)
    else
        if first(distance) > alg.threshold
            return transition
        else
            return Transition(spl)
        end
    end
end

function GpABC.simulate_distance(
    parameters::AbstractArray{Float64, 2},
    simulator_function,
    summary_statistic,
    reference_summary_statistic,
    distance_metric
)
    n_design_points = size(parameters, 1)
    y = zeros(n_design_points)
    for i in 1:n_design_points
        model_output = simulator_function(parameters[i,:])
        y[i] = distance_metric(
            summary_statistic(model_output),
            reference_summary_statistic
        )
    end
    return y
end
