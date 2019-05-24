abstract type VariationalInference <: Distribution{Multivariate, Continuous} end

abstract type VariationalObjective end

"""
    rand(vi::VariationalInference, num_samples)

Produces `num_samples` samples for the given VI method using number of samples equal to `num_samples`.
"""
function rand(vi::VariationalInference, num_samples) end

"""
    objective(vi::VariationalInference, num_samples)

Computes empirical estimates of ELBO for the given VI method using number of samples equal to `num_samples`.
"""
function objective(vo::VariationalObjective, vi::VariationalInference, model::Model, num_samples) end

"""
    optimize(vi::VariationalInference)

Finds parameters which maximizes the ELBO for the given VI method.
"""
function optimize(vo::VariationalObjective, vi::VariationalInference, model::Model) end

"""
    grad(vo::VariationalObjective, vi::VariationalInference)

Computes the gradients used in `optimize`.

Variance reduction techniques, e.g. control variates, should be implemented in this function.
"""
function grad(vo::VariationalObjective, vi::VariationalInference, model::Model) end

# objectives
include("objectives.jl")

# (::VariationalObjective)(vi::VariationalInference, model::Model, num_samples, args...; kwargs...) = begin
# end

# VI algorithms
include("advi.jl")
