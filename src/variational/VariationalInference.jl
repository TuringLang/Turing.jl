
module Variational

using DynamicPPL
using ADTypes
using Distributions
using LinearAlgebra
using LogDensityProblems
using Random
using UnicodePlots

import ..Turing: DEFAULT_ADTYPE, PROGRESS

import AdvancedVI
import Bijectors

# Reexports
using AdvancedVI: RepGradELBO, ScoreGradELBO, DoG, DoWG
export vi, RepGradELBO, ScoreGradELBO, DoG, DoWG

export meanfield_gaussian, fullrank_gaussian

include("bijectors.jl")

function make_logdensity(model::DynamicPPL.Model)
    weight = 1.0
    ctx = DynamicPPL.MiniBatchContext(DynamicPPL.DefaultContext(), weight)
    return DynamicPPL.LogDensityFunction(model, DynamicPPL.VarInfo(model), ctx)
end

function initialize_gaussian_scale(
    rng::Random.AbstractRNG,
    model::DynamicPPL.Model,
    location::AbstractVector,
    scale::AbstractMatrix;
    num_samples::Int=10,
    num_max_trials::Int=10,
    reduce_factor=one(eltype(scale)) / 2,
)
    prob = make_logdensity(model)
    ℓπ = Base.Fix1(LogDensityProblems.logdensity, prob)
    varinfo = DynamicPPL.VarInfo(model)

    n_trial = 0
    while true
        q = AdvancedVI.MvLocationScale(location, scale, Normal())
        b = Bijectors.bijector(model; varinfo=varinfo)
        q_trans = Bijectors.transformed(q, Bijectors.inverse(b))
        energy = mean(ℓπ, eachcol(rand(rng, q_trans, num_samples)))

        if isfinite(energy)
            return scale
        elseif n_trial == num_max_trials
            error("Could not find an initial")
        end

        scale = reduce_factor * scale
        n_trial += 1
    end
end

function meanfield_gaussian(
    rng::Random.AbstractRNG,
    model::DynamicPPL.Model,
    location::Union{Nothing,<:AbstractVector}=nothing,
    scale::Union{Nothing,<:Diagonal}=nothing;
    kwargs...,
)
    varinfo = DynamicPPL.VarInfo(model)
    # Use linked `varinfo` to determine the correct number of parameters.
    # TODO: Replace with `length` once this is implemented for `VarInfo`.
    varinfo_linked = DynamicPPL.link(varinfo, model)
    num_params = length(varinfo_linked[:])

    μ = if isnothing(location)
        zeros(num_params)
    else
        @assert length(location) == num_params "Length of the provided location vector, $(length(location)), does not match dimension of the target distribution, $(num_params)."
        location
    end

    L = if isnothing(scale)
        initialize_gaussian_scale(rng, model, μ, Diagonal(ones(num_params)); kwargs...)
    else
        @assert size(scale) == (num_params, num_params) "Dimensions of the provided scale matrix, $(size(scale)), does not match the dimension of the target distribution, $(num_params)."
        L = scale
    end

    q = AdvancedVI.MeanFieldGaussian(μ, L)
    b = Bijectors.bijector(model; varinfo=varinfo)
    return Bijectors.transformed(q, Bijectors.inverse(b))
end

function meanfield_gaussian(
    model::DynamicPPL.Model,
    location::Union{Nothing,<:AbstractVector}=nothing,
    scale::Union{Nothing,<:Diagonal}=nothing;
    kwargs...,
)
    return meanfield_gaussian(Random.default_rng(), model, location, scale; kwargs...)
end

function fullrank_gaussian(
    rng::Random.AbstractRNG,
    model::DynamicPPL.Model,
    location::Union{Nothing, <:AbstractVector} = nothing,
    scale::Union{Nothing, <:LowerTriangular} = nothing;
    kwargs...
)
    varinfo = DynamicPPL.VarInfo(model)
    # Use linked `varinfo` to determine the correct number of parameters.
    # TODO: Replace with `length` once this is implemented for `VarInfo`.
    varinfo_linked = DynamicPPL.link(varinfo, model)
    num_params = length(varinfo_linked[:])

    μ = if isnothing(location)
        zeros(num_params)
    else
        @assert length(location) == num_params "Length of the provided location vector, $(length(location)), does not match dimension of the target distribution, $(num_params)."
        location
    end

    L = if isnothing(scale)
        L0 = LowerTriangular(Matrix{Float64}(I, num_params, num_params))
        initialize_gaussian_scale(rng, model, μ, L0; kwargs...)
    else
        @assert size(scale) == (num_params, num_params) "Dimensions of the provided scale matrix, $(size(scale)), does not match the dimension of the target distribution, $(num_params)."
        scale
    end

    q = AdvancedVI.FullRankGaussian(μ, L)
    b = Bijectors.bijector(model; varinfo=varinfo)
    return Bijectors.transformed(q, Bijectors.inverse(b))
end

function fullrank_gaussian(
    model::DynamicPPL.Model,
    location::Union{Nothing,<:AbstractVector}=nothing,
    scale::Union{Nothing,<:Diagonal}=nothing;
    kwargs...,
)
    return fullrank_gaussian(Random.default_rng(), model, location, scale; kwargs...)
end

function vi(
    model::DynamicPPL.Model,
    q::Bijectors.TransformedDistribution,
    n_iterations::Int;
    objective=RepGradELBO(10; entropy=AdvancedVI.ClosedFormEntropyZeroGradient()),
    show_progress::Bool=PROGRESS[],
    optimizer=AdvancedVI.DoWG(),
    averager=AdvancedVI.PolynomialAveraging(),
    operator=AdvancedVI.ProximalLocationScaleEntropy(),
    adtype::ADTypes.AbstractADType=DEFAULT_ADTYPE,
    kwargs...
)
    return AdvancedVI.optimize(
        make_logdensity(model),
        objective,
        q,
        n_iterations;
        show_progress=show_progress,
        adtype,
        optimizer,
        averager,
        operator,
        kwargs...
    )
end

end
