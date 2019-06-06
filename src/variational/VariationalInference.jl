module Variational

using ..Core, ..Core.RandomVariables, ..Utilities
using Distributions, Bijectors
using ProgressMeter, LinearAlgebra
using ..Turing: PROGRESS
using ..Turing: Model, SampleFromPrior, SampleFromUniform
using ..Turing: Turing
using Random: AbstractRNG

using ForwardDiff
using Flux.Tracker, Flux.Optimise

import ..Core: getchunksize, getADtype

export
    vi,
    ADVI,
    ELBO


abstract type VariationalInference{AD} end

getchunksize(::T) where {T <: VariationalInference} = getchunksize(T)
getchunksize(::Type{<:VariationalInference{AD}}) where AD = getchunksize(AD)
getADtype(alg::VariationalInference) = getADtype(typeof(alg))
getADtype(::Type{<: VariationalInference{AD}}) where {AD} = AD

abstract type VariationalObjective end

const VariationalPosterior = Distribution{Multivariate, Continuous}

"""
    rand(vi::VariationalInference, num_samples)

Produces `num_samples` samples for the given VI method using number of samples equal to `num_samples`.
"""
function rand(vi::VariationalPosterior, num_samples) end

"""
    objective(vi::VariationalInference, num_samples)

Computes empirical estimates of ELBO for the given VI method using number of samples equal to `num_samples`.
"""
function objective(vo::VariationalObjective, vi::VariationalInference, q::VariationalPosterior, model::Model, num_samples) end

# (::VariationalObjective)(vi::VariationalInference, model::Model, num_samples, args...; kwargs...) = begin
# end

"""
    optimize(vo::VariationalObjective, vi::VariationalInference, q::VariationalPosterior, model::Model)

Finds parameters which maximizes the ELBO for the given VI method.
"""
function optimize(vo::VariationalObjective, vi::VariationalInference, q::VariationalPosterior, model::Model) end

"""
    grad(vo::VariationalObjective, vi::VariationalInference)

Computes the gradients used in `optimize`.

Variance reduction techniques, e.g. control variates, should be implemented in this function.
"""
function grad(vo::VariationalObjective, vi::VariationalInference, q::VariationalPosterior, model::Model) end

"""
    vi(model::Model, alg::VariationalInference)
    vi(model::Model, alg::VariationalInference, q::VariationalPosterior)

Constructs the variational posterior from the `model` using ``
"""
function vi(model::Model, alg::VariationalInference) end

# default implementations
function grad!(vo::VariationalObjective, alg::VariationalInference{AD}, q, model::Model, θ::AbstractVector{T}, out::DiffResults.MutableDiffResult, args...) where {T <: Real, AD <: ForwardDiffAD}
    # TODO: this probably slows down executation quite a bit; exists a better way of doing this?
    f(θ_) = - vo(alg, q, model, θ_, args...)

    chunk_size = getchunksize(alg)
    # Set chunk size and do ForwardMode.
    chunk = ForwardDiff.Chunk(min(length(θ), chunk_size))
    config = ForwardDiff.GradientConfig(f, θ, chunk)
    ForwardDiff.gradient!(out, f, θ, config)
end

# TODO: implement for `Tracker`
# function grad(vo::ELBO, alg::ADVI, q::MeanField, model::Model, f, autodiff::Val{:backward})
#     vo_tracked, vo_pullback = Tracker.forward()
# end
function grad!(vo::VariationalObjective, alg::VariationalInference{AD}, q, model::Model, θ::AbstractVector{T}, out::DiffResults.MutableDiffResult, args...) where {T <: Real, AD <: TrackerAD}
    θ_tracked = Tracker.param(θ)
    y = - vo(alg, q, model, θ_tracked, args...)
    Tracker.back!(y, 1.0)

    DiffResults.value!(out, Tracker.data(y))
    DiffResults.gradient!(out, Tracker.grad(θ_tracked))
end

# distributions
include("distributions.jl")

# objectives
include("objectives.jl")

# VI algorithms
include("advi.jl")

end
