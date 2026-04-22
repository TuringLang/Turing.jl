
module Variational

using AdvancedVI:
    AdvancedVI,
    KLMinRepGradDescent,
    KLMinRepGradProxDescent,
    KLMinScoreGradDescent,
    KLMinWassFwdBwd,
    KLMinNaturalGradDescent,
    KLMinSqrtNaturalGradDescent,
    FisherMinBatchMatch

using ADTypes
using Bijectors: Bijectors
using Distributions
using DynamicPPL: DynamicPPL, LogDensityFunction
using LinearAlgebra
using LogDensityProblems: LogDensityProblems
using Random
using ..Turing: DEFAULT_ADTYPE, PROGRESS

export vi,
    q_locationscale,
    q_meanfield_gaussian,
    q_fullrank_gaussian,
    KLMinRepGradProxDescent,
    KLMinRepGradDescent,
    KLMinScoreGradDescent,
    KLMinWassFwdBwd,
    KLMinNaturalGradDescent,
    KLMinSqrtNaturalGradDescent,
    FisherMinBatchMatch

requires_unconstrained_space(::AdvancedVI.AbstractVariationalAlgorithm) = true
requires_unconstrained_space(::AdvancedVI.KLMinRepGradProxDescent) = true
requires_unconstrained_space(::AdvancedVI.KLMinRepGradDescent) = true
requires_unconstrained_space(::AdvancedVI.KLMinScoreGradDescent) = false
requires_unconstrained_space(::AdvancedVI.KLMinWassFwdBwd) = true
requires_unconstrained_space(::AdvancedVI.KLMinNaturalGradDescent) = true
requires_unconstrained_space(::AdvancedVI.KLMinSqrtNaturalGradDescent) = true
requires_unconstrained_space(::AdvancedVI.FisherMinBatchMatch) = true

"""
    q_initialize_scale(
        rng::Random.AbstractRNG,
        ldf::DynamicPPL.LogDensityFunction,
        location::AbstractVector,
        scale::AbstractMatrix,
        basedist::Distributions.UnivariateDistribution;
        num_samples::Int = 10,
        num_max_trials::Int = 10,
        reduce_factor::Real = one(eltype(scale)) / 2
    )

Given an initial location-scale distribution `q` formed by `location`, `scale`, and `basedist`, shrink `scale` until the expectation of log-densities of `ldf` taken over `q` are finite.
If the log-densities are not finite even after `num_max_trials`, throw an error.

For reference, a location-scale distribution \$q\$ formed by `location`, `scale`, and `basedist` is a distribution where its sampling process \$z \\sim q\$ can be represented as
```julia
u = rand(basedist, d)
z = scale * u + location
```

# Arguments
- `ldf`: The target log-density function.
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
    ldf::LogDensityFunction,
    location::AbstractVector,
    scale::AbstractMatrix,
    basedist::Distributions.UnivariateDistribution;
    num_samples::Int=10,
    num_max_trials::Int=10,
    reduce_factor::Real=one(eltype(scale)) / 2,
    kwargs..., # must take extra kwargs even if they are ignored
)
    num_max_trials > 0 || error("num_max_trials must be a positive integer")
    n_trial = 0
    while true
        q = AdvancedVI.MvLocationScale(location, scale, basedist)
        energy = mean(
            map(1:num_samples) do _
                z = rand(rng, q)
                LogDensityProblems.logdensity(ldf, z)
            end,
        )

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
        ldf::DynamicPPL.LogDensityFunction;
        location::Union{Nothing,<:AbstractVector} = nothing,
        scale::Union{Nothing,<:Diagonal,<:LowerTriangular} = nothing,
        meanfield::Bool = true,
        basedist::Distributions.UnivariateDistribution = Normal()
    )

Find a numerically non-degenerate variational distribution `q` for approximating the target `LogDensityFunction` within the location-scale variational family formed by the type of `scale` and `basedist`.

The distribution can be manually specified by setting `location`, `scale`, and `basedist`.
Otherwise, it chooses a Gaussian with zero-mean and scale `0.6*I` (covariance of `0.6^2*I`) by default.
This guarantees that the samples from the initial variational approximation will fall in the range of (-2, 2) with 99.9% probability, which mimics the behavior of the `Turing.InitFromUniform()` strategy.

Whether the default choice is used or not, the `scale` may be adjusted via `q_initialize_scale` so that the log-densities of `model` are finite over the samples from `q`.
If `meanfield` is set as `true`, the scale of `q` is restricted to be a diagonal matrix and only the diagonal of `scale` is used.

For reference, a location-scale distribution \$q\$ formed by `location`, `scale`, and `basedist` is a distribution where its sampling process \$z \\sim q\$ can be represented as
```julia
u = rand(basedist, d)
z = scale * u + location
```

# Arguments
- `ldf`: The target log-density function.

# Keyword Arguments
- `location`: The location parameter of the initialization. If `nothing`, a vector of zeros is used.
- `scale`: The scale parameter of the initialization. If `nothing`, an identity matrix is used.
- `meanfield`: Whether to use the mean-field approximation. If `true`, `scale` is converted into a `Diagonal` matrix. Otherwise, it is converted into a `LowerTriangular` matrix.
- `basedist`: The base distribution of the location-scale family.

The remaining keywords are passed to `q_initialize_scale`.

# Returns 
- An `AdvancedVI.LocationScale` distribution matching the support of `ldf`.
"""
function q_locationscale(
    rng::Random.AbstractRNG,
    ldf::LogDensityFunction;
    location::Union{Nothing,<:AbstractVector}=nothing,
    scale::Union{Nothing,<:Diagonal,<:LowerTriangular}=nothing,
    meanfield::Bool=true,
    basedist::Distributions.UnivariateDistribution=Normal(),
    kwargs...,
)
    num_params = LogDensityProblems.dimension(ldf)

    μ = if isnothing(location)
        zeros(num_params)
    else
        @assert length(location) == num_params "Length of the provided location vector, $(length(location)), does not match dimension of the target distribution, $(num_params)."
        location
    end

    L = if isnothing(scale)
        if meanfield
            q_initialize_scale(
                rng, ldf, μ, Diagonal(fill(0.6, num_params)), basedist; kwargs...
            )
        else
            L0 = LowerTriangular(Matrix{Float64}(0.6 * I, num_params, num_params))
            q_initialize_scale(rng, ldf, μ, L0, basedist; kwargs...)
        end
    else
        @assert size(scale) == (num_params, num_params) "Dimensions of the provided scale matrix, $(size(scale)), does not match the dimension of the target distribution, $(num_params)."
        if meanfield
            Diagonal(diag(scale))
        else
            LowerTriangular(Matrix(scale))
        end
    end
    return AdvancedVI.MvLocationScale(μ, L, basedist)
end
function q_locationscale(ldf::LogDensityFunction; kwargs...)
    return q_locationscale(Random.default_rng(), ldf; kwargs...)
end

"""
    q_meanfield_gaussian(
        [rng::Random.AbstractRNG,]
        ldf::DynamicPPL.LogDensityFunction;
        location::Union{Nothing,<:AbstractVector} = nothing,
        scale::Union{Nothing,<:Diagonal} = nothing,
        kwargs...
    )

Find a numerically non-degenerate mean-field Gaussian `q` for approximating the target `ldf::LogDensityFunction`.

If the `scale` set as `nothing`, the default value will be a zero-mean Gaussian with a `Diagonal` scale matrix (the "mean-field" approximation) no larger than `0.6*I` (covariance of `0.6^2*I`).
This guarantees that the samples from the initial variational approximation will fall in the range of (-2, 2) with 99.9% probability, which mimics the behavior of the `Turing.InitFromUniform()` strategy.
Whether the default choice is used or not, the `scale` may be adjusted via `q_initialize_scale` so that the log-densities of `model` are finite over the samples from `q`.

# Arguments
- `ldf`: The target log-density function.

# Keyword Arguments
- `location`: The location parameter of the initialization. If `nothing`, a vector of zeros is used.
- `scale`: The scale parameter of the initialization. If `nothing`, an identity matrix is used.

The remaining keyword arguments are passed to `q_locationscale`.

# Returns 
- An `AdvancedVI.LocationScale` distribution matching the support of `ldf`.
"""
function q_meanfield_gaussian(
    rng::Random.AbstractRNG,
    ldf::LogDensityFunction;
    location::Union{Nothing,<:AbstractVector}=nothing,
    scale::Union{Nothing,<:Diagonal}=nothing,
    kwargs...,
)
    return q_locationscale(
        rng, ldf; location, scale, meanfield=true, basedist=Normal(), kwargs...
    )
end
function q_meanfield_gaussian(ldf::LogDensityFunction; kwargs...)
    return q_meanfield_gaussian(Random.default_rng(), ldf; kwargs...)
end

"""
    q_fullrank_gaussian(
        [rng::Random.AbstractRNG,]
        ldf::DynamicPPL.LogDensityFunction;
        location::Union{Nothing,<:AbstractVector} = nothing,
        scale::Union{Nothing,<:LowerTriangular} = nothing,
        kwargs...
    )

Find a numerically non-degenerate Gaussian `q` with a scale with full-rank factors (traditionally referred to as a "full-rank family") for approximating the target `ldf::LogDensityFunction`.

If the `scale` set as `nothing`, the default value will be a zero-mean Gaussian with a `LowerTriangular` scale matrix (resulting in a covariance with "full-rank" factors) no larger than `0.6*I` (covariance of `0.6^2*I`).
This guarantees that the samples from the initial variational approximation will fall in the range of (-2, 2) with 99.9% probability, which mimics the behavior of the `Turing.InitFromUniform()` strategy.
Whether the default choice is used or not, the `scale` may be adjusted via `q_initialize_scale` so that the log-densities of `model` are finite over the samples from `q`.

# Arguments
- `ldf`: The target log-density function.

# Keyword Arguments
- `location`: The location parameter of the initialization. If `nothing`, a vector of zeros is used.
- `scale`: The scale parameter of the initialization. If `nothing`, an identity matrix is used.

The remaining keyword arguments are passed to `q_locationscale`.

# Returns 
- An `AdvancedVI.LocationScale` distribution matching the support of `ldf`.
"""
function q_fullrank_gaussian(
    rng::Random.AbstractRNG,
    ldf::LogDensityFunction;
    location::Union{Nothing,<:AbstractVector}=nothing,
    scale::Union{Nothing,<:LowerTriangular}=nothing,
    kwargs...,
)
    return q_locationscale(
        rng, ldf; location, scale, meanfield=false, basedist=Normal(), kwargs...
    )
end
function q_fullrank_gaussian(ldf::LogDensityFunction; kwargs...)
    return q_fullrank_gaussian(Random.default_rng(), ldf; kwargs...)
end

"""
    VIResult(ldf, q, info, state)

- `ldf`: A [`DynamicPPL.LogDensityFunction`](@extref) corresponding to the target model (the
  original model can be accessed as `ldf.model`). If the VI process was run in unconstrained
  space, this LogDensityFunction will also be in unconstrained space.
- `q`: Output variational distribution of `algorithm`. Note that, as above, this will
  typically also be in unconstrained space.
- `state`: Collection of states used by `algorithm`. This can be used to resume from a past
  call to `vi`.
- `info`: Information generated while executing `algorithm`.
"""
struct VIResult{L<:LogDensityFunction,Q<:Distribution,I<:AbstractArray{<:NamedTuple},S}
    ldf::L
    q::Q
    info::I
    state::S
end

function Base.show(io::IO, ::MIME"text/plain", r::VIResult)
    printstyled(io, "VIResult\n"; bold=true)
    println(io, "  ├ q    : $(nameof(typeof(r.q)))")
    n_iters = length(r.info)
    println(io, "  ├ info : $(length(r.info))-element $(typeof(r.info))")
    if n_iters > 0
        println(io, "  │        final iteration:")
        last_info = r.info[end]
        for (i, (k, v)) in enumerate(pairs(last_info))
            tree_char = i == length(last_info) ? "└" : "├"
            println(io, "  │         $(tree_char) $k = $v")
        end
    end
    print(io, "  └ (2 more fields: state, ldf)")
    return nothing
end

"""
    Base.rand(rng::Random.AbstractRNG, res::VIResult, sz...)

Draw a sample, or array of samples, from the variational distribution `q` in `res`. Each
sample is a [`DynamicPPL.VarNamedTuple`](@extref DynamicPPL.VarNamedTuples.VarNamedTuple)
containing raw parameter values.
"""
function Base.rand(rng::Random.AbstractRNG, res::VIResult, sz::Integer...)
    # TODO(penelopeysm): Should we expose a way to get colon_eq results as well -- maybe a
    # kwarg?
    function to_vnt(v::AbstractVector)
        pws = DynamicPPL.ParamsWithStats(
            v, res.ldf; include_colon_eq=false, include_log_probs=false
        )
        return pws.params
    end
    if sz == ()
        return to_vnt(rand(rng, res.q))
    else
        # re. stack: https://github.com/TuringLang/AdvancedVI.jl/issues/245
        x = stack(rand(rng, res.q, sz...))
        return map(to_vnt, eachslice(x; dims=ntuple(i -> i + 1, length(sz))))
    end
end
Base.rand(res::VIResult, sz::Integer...) = Base.rand(Random.default_rng(), res, sz...)

"""
    vi(
        [rng::Random.AbstractRNG,]
        model::DynamicPPL.Model,
        family,
        max_iter::Int;
        adtype::ADTypes.AbstractADType=DEFAULT_ADTYPE,
        algorithm::AdvancedVI.AbstractVariationalAlgorithm = KLMinRepGradProxDescent(
            adtype; n_samples=10
        ),
        unconstrained::Bool=requires_unconstrained_space(algorithm),
        fix_transforms::Bool=false,
        show_progress::Bool = Turing.PROGRESS[],
        kwargs...
    )

Approximate the target `model` via the variational inference algorithm `algorithm` using a variational family specified by `family`.
This is a thin wrapper around `AdvancedVI.optimize`.

The default `algorithm`, `KLMinRepGradProxDescent` ([relevant docs](https://turinglang.org/AdvancedVI.jl/dev/klminrepgradproxdescent/)), assumes `family` returns a `AdvancedVI.MvLocationScale`, which is true if `family` is `q_fullrank_gaussian` or `q_meanfield_gaussian`.
For other variational families, refer to the documentation of `AdvancedVI` to determine the best algorithm and other options.

# Arguments
- `model`: The target `DynamicPPL.Model`.
- `family`: A function which is used to generate an initial variational approximation.
  Existing choices in Turing are [`q_locationscale`](@ref), [`q_meanfield_gaussian`](@ref), and [`q_fullrank_gaussian`](@ref).
- `max_iter`: Maximum number of steps.
- Any additional arguments are passed on to `AdvancedVI.optimize`.

# Keyword Arguments
- `adtype`: Automatic differentiation backend to be applied to the log-density. The default value for `algorithm` also uses this backend for differentiating the variational objective.
- `algorithm`: Variational inference algorithm. The default is `KLMinRepGradProxDescent`, please refer to [AdvancedVI docs](https://turinglang.org/AdvancedVI.jl/stable/) for all the options.
- `show_progress`: Whether to show the progress bar.
- `unconstrained`: Whether to transform the posterior to be unconstrained for running the variational inference algorithm. The default value depends on the chosen `algorithm` (most algorithms require unconstrained space).
- `fix_transforms`: Whether to precompute the transforms needed to convert model parameters to (possibly unconstrained) vectors. This can lead to performance improvements, but if any transforms depend on model parameters, setting `fix_transforms=true` can silently yield incorrect results.
- Any additional keyword arguments are passed on both to the function `initial_approx`, and also to `AdvancedVI.optimize`.


See the docs of `AdvancedVI.optimize` for additional keyword arguments.

# Returns 

A [`VIResult`](@ref) object: please see its docstring for information.
"""
function vi(
    rng::Random.AbstractRNG,
    model::DynamicPPL.Model,
    family,
    max_iter::Int,
    args...;
    adtype::ADTypes.AbstractADType=DEFAULT_ADTYPE,
    algorithm::AdvancedVI.AbstractVariationalAlgorithm=KLMinRepGradProxDescent(
        adtype; n_samples=10
    ),
    unconstrained::Bool=requires_unconstrained_space(algorithm),
    fix_transforms::Bool=false,
    show_progress::Bool=PROGRESS[],
    kwargs...,
)
    transform_strategy = unconstrained ? DynamicPPL.LinkAll() : DynamicPPL.UnlinkAll()
    prob = LogDensityFunction(
        model, DynamicPPL.getlogjoint_internal, transform_strategy; adtype, fix_transforms
    )
    q = family(rng, prob; kwargs...)
    q, info, state = AdvancedVI.optimize(
        rng, algorithm, max_iter, prob, q, args...; show_progress=show_progress, kwargs...
    )
    return VIResult(prob, q, info, state)
end

function vi(model::DynamicPPL.Model, family, max_iter::Int; kwargs...)
    return vi(Random.default_rng(), model, family, max_iter; kwargs...)
end

end
