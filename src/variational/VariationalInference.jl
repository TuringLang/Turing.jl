module Variational

using ..Core, ..Core.RandomVariables, ..Utilities
using Distributions, Bijectors
using ProgressMeter, LinearAlgebra
using ..Turing: PROGRESS
using ..Turing: Model, SampleFromPrior, SampleFromUniform
using ..Turing: Turing
using Random: AbstractRNG

export
    vi,
    ADVI,
    ELBO

abstract type VariationalInference end

abstract type VariationalObjective end

abstract type VariationalPosterior <: Distribution{Multivariate, Continuous} end

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

"""
    optimize(vi::VariationalInference)

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

# objectives
include("objectives.jl")

# (::VariationalObjective)(vi::VariationalInference, model::Model, num_samples, args...; kwargs...) = begin
# end

# VI algorithms
include("advi.jl")

end
