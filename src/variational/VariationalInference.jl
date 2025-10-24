
module Variational

using AdvancedVI:
    AdvancedVI, KLMinRepGradDescent, KLMinRepGradProxDescent, KLMinScoreGradDescent
using ADTypes
using Bijectors: Bijectors
using Distributions
using DynamicPPL
using LinearAlgebra
using LogDensityProblems
using Random
using ..Turing: DEFAULT_ADTYPE, PROGRESS

export vi,
    q_locationscale,
    q_meanfield_gaussian,
    q_fullrank_gaussian,
    KLMinRepGradProxDescent,
    KLMinRepGradDescent,
    KLMinScoreGradDescent

"""
    q_initialize_scale(
        [rng::Random.AbstractRNG,]
        model::DynamicPPL.Model,
        location::AbstractVector,
        scale::AbstractMatrix,
        basedist::Distributions.UnivariateDistribution;
        num_samples::Int = 10,
        num_max_trials::Int = 10,
        reduce_factor::Real = one(eltype(scale)) / 2
    )

Given an initial location-scale distribution `q` formed by `location`, `scale`, and `basedist`, shrink `scale` until the expectation of log-densities of `model` taken over `q` are finite.
If the log-densities are not finite even after `num_max_trials`, throw an error.

For reference, a location-scale distribution \$q\$ formed by `location`, `scale`, and `basedist` is a distribution where its sampling process \$z \\sim q\$ can be represented as
```julia
u = rand(basedist, d)
z = scale * u + location
```

# Arguments
- `model`: The target `DynamicPPL.Model`.
- `location`: The location parameter of the initialization.
- `scale`: The scale parameter of the initialization.
- `basedist`: The base distribution of the location-scale family.

# Keyword Arguments
- `num_samples`: Number of samples used to compute the average log-density at each trial.
- `num_max_trials`: Number of trials until throwing an error.
- `reduce_factor`: Factor for shrinking the scale. After `n` trials, the scale is then `scale*reduce_factor^n`.

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
    prob = LogDensityFunction(model)
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
    q_locationscale(
        [rng::Random.AbstractRNG,]
        model::DynamicPPL.Model;
        location::Union{Nothing,<:AbstractVector} = nothing,
        scale::Union{Nothing,<:Diagonal,<:LowerTriangular} = nothing,
        meanfield::Bool = true,
        basedist::Distributions.UnivariateDistribution = Normal()
    )

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
- `model`: The target `DynamicPPL.Model`.

# Keyword Arguments
- `location`: The location parameter of the initialization. If `nothing`, a vector of zeros is used.
- `scale`: The scale parameter of the initialization. If `nothing`, an identity matrix is used.
- `meanfield`: Whether to use the mean-field approximation. If `true`, `scale` is converted into a `Diagonal` matrix. Otherwise, it is converted into a `LowerTriangular` matrix.
- `basedist`: The base distribution of the location-scale family.

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
            LowerTriangular(Matrix(scale))
        end
    end
    q = AdvancedVI.MvLocationScale(μ, L, basedist)
    b = Bijectors.bijector(model; varinfo=varinfo)
    return Bijectors.transformed(q, Bijectors.inverse(b))
end

function q_locationscale(model::DynamicPPL.Model; kwargs...)
    return q_locationscale(Random.default_rng(), model; kwargs...)
end

"""
    q_meanfield_gaussian(
        [rng::Random.AbstractRNG,]
        model::DynamicPPL.Model;
        location::Union{Nothing,<:AbstractVector} = nothing,
        scale::Union{Nothing,<:Diagonal} = nothing,
        kwargs...
    )

Find a numerically non-degenerate mean-field Gaussian `q` for approximating the  target `model`.

# Arguments
- `model`: The target `DynamicPPL.Model`.

# Keyword Arguments
- `location`: The location parameter of the initialization. If `nothing`, a vector of zeros is used.
- `scale`: The scale parameter of the initialization. If `nothing`, an identity matrix is used.

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
    q_fullrank_gaussian(
        [rng::Random.AbstractRNG,]
        model::DynamicPPL.Model;
        location::Union{Nothing,<:AbstractVector} = nothing,
        scale::Union{Nothing,<:LowerTriangular} = nothing,
        kwargs...
    )

Find a numerically non-degenerate Gaussian `q` with a scale with full-rank factors (traditionally referred to as a "full-rank family") for approximating the target `model`.

# Arguments
- `model`: The target `DynamicPPL.Model`.

# Keyword Arguments
- `location`: The location parameter of the initialization. If `nothing`, a vector of zeros is used.
- `scale`: The scale parameter of the initialization. If `nothing`, an identity matrix is used.

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
    vi(
        [rng::Random.AbstractRNG,]
        model::DynamicPPL.Model,
        q,
        max_iter::Int;
        adtype::ADTypes.AbstractADType=DEFAULT_ADTYPE,
        algorithm::AdvancedVI.AbstractVariationalAlgorithm = KLMinRepGradProxDescent(
            adtype; n_samples=10
        ),
        show_progress::Bool = Turing.PROGRESS[],
        kwargs...
    )

Approximate the target `model` via the variational inference algorithm `algorithm` by starting from the initial variational approximation `q`.
This is a thin wrapper around `AdvancedVI.optimize`.
The default `algorithm`, `KLMinRepGradProxDescent` ([relevant docs](https://turinglang.org/AdvancedVI.jl/dev/klminrepgradproxdescent/)), assumes `q` uses `AdvancedVI.MvLocationScale`, which can be constructed by invoking `q_fullrank_gaussian` or `q_meanfield_gaussian`.
For other variational families, refer to `AdvancedVI` to determine the best algorithm and options.

# Arguments
- `model`: The target `DynamicPPL.Model`.
- `q`: The initial variational approximation.
- `max_iter`: Maximum number of steps.

# Keyword Arguments
- `adtype`: Automatic differentiation backend to be applied to the log-density. The default value for `algorithm` also uses this backend for differentiation the variational objective.
- `algorithm`: Variational inference algorithm.
- `show_progress`: Whether to show the progress bar.

See the docs of `AdvancedVI.optimize` for additional keyword arguments.

# Returns 
- `q`: Output variational distribution of `algorithm`.
- `state`: Collection of states used by `algorithm`. This can be used to resume from a past call to `vi`.
- `info`: Information generated while executing `algorithm`.
"""
function vi(
    rng::Random.AbstractRNG,
    model::DynamicPPL.Model,
    q,
    max_iter::Int,
    args...;
    adtype::ADTypes.AbstractADType=DEFAULT_ADTYPE,
    algorithm::AdvancedVI.AbstractVariationalAlgorithm=KLMinRepGradProxDescent(
        adtype; n_samples=10
    ),
    show_progress::Bool=PROGRESS[],
    kwargs...,
)
    return AdvancedVI.optimize(
        rng,
        algorithm,
        max_iter,
        LogDensityFunction(model; adtype),
        q,
        args...;
        show_progress=show_progress,
        kwargs...,
    )
end

function vi(model::DynamicPPL.Model, q, max_iter::Int; kwargs...)
    return vi(Random.default_rng(), model, q, max_iter; kwargs...)
end

end
