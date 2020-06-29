using GpABC

import .Inference: AbstractABC, AbstractSamplerState, Model, AbstractRNG, Transition


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

function GPABCState(alg::GPABC, model::Model)
    vi = Turing.VarInfo(model)

    observations = Turing.Inference.get_data(model)
    proposed = similar(observations)
    empty!(proposed)

    reference_summary_statistic = alg.summary_statistic(reshape(observations, (:, 1)))

    return GPABCState(vi, proposed, reference_summary_statistic)
end

DynamicPPL.Sampler(alg::GPABC, model::Model, s::DynamicPPL.Selector) = DynamicPPL.Sampler(alg, Dict{Symbol, Any}(), s, GPABCState(alg, model))

function AbstractMCMC.step!(
    rng::AbstractRNG,
    model::Model,
    spl::DynamicPPL.Sampler{<:GPABC},
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
