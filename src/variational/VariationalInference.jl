
module Variational

using DynamicPPL
using ADTypes
using Distributions
using LinearAlgebra
using LogDensityProblems
using Random

import ..Turing: DEFAULT_ADTYPE, PROGRESS

import AdvancedVI
import Bijectors

export vi, q_locationscale, q_meanfield_gaussian, q_fullrank_gaussian

include("deprecated.jl")

function make_logdensity(model::DynamicPPL.Model)
    weight = 1.0
    ctx = DynamicPPL.MiniBatchContext(DynamicPPL.DefaultContext(), weight)
    return DynamicPPL.LogDensityFunction(model, DynamicPPL.VarInfo(model), ctx)
end

"""
    q_initialize_scale([rng, ]model, location, scale, basedist; num_samples, num_max_trials, reduce_factor)

Given an initial location-scale distribution `q` formed by `location`, `scale`, and `basedist`, shrink `scale` until the expectation of log-densities of `model` taken over `q` are finite.
If the log-densities are not finite even after `num_max_trials`, throw an error.

For reference, a location-scale distribution \$q\$ formed by `location`, `scale`, and `basedist` is a distribution where its sampling process \$z \\sim q\$ can be represented as
```julia
u = rand(basedist, d)
z = scale * u + location
```

# Arguments
- `model::DynamicPPL.Model`: The target `DynamicPPL.Model`.
- `location::AbstractVector`: The location parameter of the initialization.
- `scale::AbstractMatrix`: The scale parameter of the initialization.
- `basedist::Distributions.UnivariateDistribution`: The base distribution of the location-scale family.

# Keyword Arguments
- `num_samples::Int`: Number of samples used to compute the average log-density at each trial. (Default: `10`.)
- `num_max_trials::Int`: Number of trials until throwing an error. (Default: `10`.)
- `reduce_factor::Real`: Factor for shrinking the scale. After `n` trials, the scale is then `scale*reduce_factor^n`. (Default: `0.5`.)

# Returns 
- `scale_adj`: The adjusted scale matrix matching the type of `scale`.
"""
function q_initialize_scale(
    rng::Random.AbstractRNG,
    model::DynamicPPL.Model,
    location::AbstractVector,
    scale::AbstractMatrix,
    basedist::Distributions.UnivariateDistribution;
    num_samples::Int=10,
    num_max_trials::Int=10,
    reduce_factor::Real=one(eltype(scale)) / 2,
)
    prob = make_logdensity(model)
    ℓπ = Base.Fix1(LogDensityProblems.logdensity, prob)
    varinfo = DynamicPPL.VarInfo(model)

    n_trial = 0
    while true
        q = AdvancedVI.MvLocationScale(location, scale, basedist)
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

"""
    q_locationscale([rng, ]model; location, scale, meanfield, basedist)

Find a numerically non-degenerate variational distribution `q` for approximating the  target `model` within the location-scale variational family formed by the type of `scale` and `basedist`.

The distribution can be manually specified by setting `location`, `scale`, and `basedist`.
Otherwise, it chooses a standard Gaussian by default.
Whether the default choice is used or not, the `scale` may be adjusted via `q_initialize_scale` so that the log-densities of `model` are finite over the samples from `q`.
If `meanfield` is set as `true`, the scale of `q` is restricted to be a diagonal matrix and only the diagonal of `scale` is used.

For reference, a location-scale distribution \$q\$ formed by `location`, `scale`, and `basedist` is a distribution where its sampling process \$z \\sim q\$ can be represented as
```julia
u = rand(basedist, d)
z = scale * u + location
```

# Arguments
- `model::DynamicPPL.Model`: The target `DynamicPPL.Model`.

# Keyword Arguments
- `location::Union{Nothing,<:AbstractVector}`: The location parameter of the initialization. If `nothing`, a vector of zeros is used.
- `scale::Union{Nothing,<:Diagonal,<:LowerTriangular}`: The scale parameter of the initialization. If `nothing`, an identity matrix is used.
- `basedist::Distributions.UnivariateDistribution`: The distribution 

The remaining keywords are passed to `q_initialize_scale`.

# Returns 
- `q::Bijectors.TransformedDistribution`: A `AdvancedVI.LocationScale` distribution matching the support of `model`.
"""
function q_locationscale(
    rng::Random.AbstractRNG,
    model::DynamicPPL.Model;
    location::Union{Nothing,<:AbstractVector}=nothing,
    scale::Union{Nothing,<:Diagonal,<:LowerTriangular}=nothing,
    meanfield::Bool=true,
    basedist::Distributions.UnivariateDistribution=Normal(),
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
        if meanfield
            q_initialize_scale(rng, model, μ, Diagonal(ones(num_params)), basedist; kwargs...)
        else
            L0 = LowerTriangular(Matrix{Float64}(I, num_params, num_params))
            q_initialize_scale(rng, model, μ, L0, basedist; kwargs...)
        end
    else
        @assert size(scale) == (num_params, num_params) "Dimensions of the provided scale matrix, $(size(scale)), does not match the dimension of the target distribution, $(num_params)."
        if meanfield
            Diagonal(diag(scale))
        else
            scale
        end
    end
    q = AdvancedVI.MvLocationScale(μ, L, basedist)
    b = Bijectors.bijector(model; varinfo=varinfo)
    return Bijectors.transformed(q, Bijectors.inverse(b))
end

"""
    q_meanfield_gaussian([rng, ]model; location, scale, kwargs...)

Find a numerically non-degenerate mean-field Gaussian `q` for approximating the  target `model`.

# Arguments
- `model::DynamicPPL.Model`: The target `DynamicPPL.Model`.

# Keyword Arguments
- `location::Union{Nothing,<:AbstractVector}`: The location parameter of the initialization. If `nothing`, a vector of zeros is used.
- `scale::Union{Nothing,<:Diagonal}`: The scale parameter of the initialization. If `nothing`, an identity matrix is used.

The remaining keyword arguments are passed to `q_locationscale`.

# Returns 
- `q::Bijectors.TransformedDistribution`: A `AdvancedVI.LocationScale` distribution matching the support of `model`.
"""
function q_meanfield_gaussian(
    rng::Random.AbstractRNG,
    model::DynamicPPL.Model;
    location::Union{Nothing,<:AbstractVector}=nothing,
    scale::Union{Nothing,<:Diagonal}=nothing,
    kwargs...,
)
    return q_locationscale(
        rng, model; location, scale, meanfield=true, basedist=Normal(), kwargs...
    )
end

function q_meanfield_gaussian(model::DynamicPPL.Model; kwargs...)
    return q_meanfield_gaussian(Random.default_rng(), model; kwargs...)
end

"""
    q_fullrank_gaussian([rng, ]model; location, scale, kwargs...)

Find a numerically non-degenerate Gaussian `q` with a dense scale (traditionally referred to as "full-rank") for approximating the target `model`.

# Arguments
- `model::DynamicPPL.Model`: The target `DynamicPPL.Model`.

# Keyword Arguments
- `location::Union{Nothing,<:AbstractVector}`: The location parameter of the initialization. If `nothing`, a vector of zeros is used.
- `scale::Union{Nothing,<:LowerTriangular}`: The scale parameter of the initialization. If `nothing`, an identity matrix is used.

The remaining keyword arguments are passed to `q_locationscale`.

# Returns 
- `q::Bijectors.TransformedDistribution`: A `AdvancedVI.LocationScale` distribution matching the support of `model`.
"""
function q_fullrank_gaussian(
    rng::Random.AbstractRNG,
    model::DynamicPPL.Model;
    location::Union{Nothing,<:AbstractVector}=nothing,
    scale::Union{Nothing,<:LowerTriangular}=nothing,
    kwargs...,
)
    return q_locationscale(
        rng, model; location, scale, meanfield=false, basedist=Normal(), kwargs...
    )
end

function q_fullrank_gaussian(model::DynamicPPL.Model; kwargs...)
    return q_fullrank_gaussian(Random.default_rng(), model; kwargs...)
end

"""
    vi([rng, ]model, q, n_iterations; objective, show_progress, optimizer, averager, operator, adtype, kwargs...)

Approximating the target `model` via variational inference by optimizing `objective` with the initialization `q`.
This is a thin wrapper around `AdvancedVI.optimize`.

# Arguments
- `model::DynamicPPL.Model`: The target `DynamicPPL.Model`.
- `q`: The initial variational approximation.
- `n_iterations::Int`: Number of optimization steps.

# Keyword Arguments
- `objective::AdvancedVI.AbstractVariationalObjective`: Variational objective to be optimized.
- `show_progress::Bool`: Whether to show the progress bar. (Default: `Turing.PROGRESS[]`.)
- `optimizer::Optimisers.AbstractRule`: Optimization algorithm. (Default: `AdvancedVI.DoWG`.)
- `averager::AdvancedVI.AbstractAverager`: Parameter averaging strategy. (Default: `AdvancedVI.PolynomialAveraging()`) 
- `operator::AdvancedVI.AbstractOperator`: Operator applied after each optimization step. (Default: `AdvancedVI.ProximalLocationScaleEntropy()`.) 
- `adtype::ADTypes.AbstractADType`: Automatic differentiation backend. (Default: `Turing.DEFAULT_ADTYPE`)

See the docs of `AvancedVI.optimize` for additional keyword arguments.

# Returns 
- `q`: Variational distribution formed by the last iterate of the optimization run.
- `q_avg`: Variational distribution formed by the averaged iterates according to `averager`.
- `state`: Collection of states used for optimization. This can be used to resume from a past call to `vi`.
- `info`: Information generated during the optimization run.
"""
function vi(
    rng::Random.AbstractRNG,
    model::DynamicPPL.Model,
    q,
    n_iterations::Int;
    objective=AdvancedVI.RepGradELBO(
        10; entropy=AdvancedVI.ClosedFormEntropyZeroGradient()
    ),
    show_progress::Bool=PROGRESS[],
    optimizer=AdvancedVI.DoWG(),
    averager=AdvancedVI.PolynomialAveraging(),
    operator=AdvancedVI.ProximalLocationScaleEntropy(),
    adtype::ADTypes.AbstractADType=DEFAULT_ADTYPE,
    kwargs...,
)
    return AdvancedVI.optimize(
        rng,
        make_logdensity(model),
        objective,
        q,
        n_iterations;
        show_progress=show_progress,
        adtype,
        optimizer,
        averager,
        operator,
        kwargs...,
    )
end

function vi(model::DynamicPPL.Model, q, n_iterations::Int; kwargs...)
    return vi(Random.default_rng(), model, q, n_iterations; kwargs...)
end

end
