module Variational

using ..Core, ..Utilities
using Distributions, Bijectors, DynamicPPL
using ProgressMeter, LinearAlgebra
using ..Turing: PROGRESS
using DynamicPPL: Model, SampleFromPrior, SampleFromUniform
using ..Turing: Turing
using Random: AbstractRNG

using ForwardDiff
using Tracker

import ..Core: getchunksize, getADtype

using Requires
function __init__()
    @require Flux="587475ba-b771-5e3f-ad9e-33799f191a9c" apply!(o, x, Δ) = Flux.Optimise.apply!(o, x, Δ)
end

export
    vi,
    ADVI,
    ELBO,
    TruncatedADAGrad

abstract type VariationalInference{AD} end

getchunksize(::Type{<:VariationalInference{AD}}) where AD = getchunksize(AD)
getADtype(::VariationalInference{AD}) where AD = AD

abstract type VariationalObjective end

const VariationalPosterior = Distribution{Multivariate, Continuous}

"""
    rand(vi::VariationalInference, num_samples)

Produces `num_samples` samples for the given VI method using number of samples equal
to `num_samples`.
"""
function rand(vi::VariationalPosterior, num_samples) end

"""
    objective(vi::VariationalInference, q::VariationalPosterior, model::Model, args...)

Computes the variational objective to be optimized for a given VI method.
"""
function objective(
    vi::VariationalInference,
    q::VariationalPosterior,
    model::Model,
    num_samples)
end


"""
    grad!(vo, vi::VariationalInference, q::VariationalPosterior, model::Model, θ, out, args...)

Computes the gradients used in `optimize!`. Default implementation is provided for 
`VariationalInference{AD}` where `AD` is either `ForwardDiffAD` or `TrackerAD`.
This implicitly also gives a default implementation of `optimize!`.

Variance reduction techniques, e.g. control variates, should be implemented in this function.
"""
function grad!(
    vo, vi::VariationalInference,
    q::VariationalPosterior,
    model::Model,
    θ,
    out,
    args...
)
    error("Turing.Variational.grad!: unmanaged variational inference algorithm: "
          * "$(typeof(alg))")
end

"""
    vi(model::Model, alg::VariationalInference)
    vi(model::Model, alg::VariationalInference, q::VariationalPosterior)

Constructs the variational posterior from the `model` and performs the optimization
following the configuration of the given `VariationalInference` instance.
"""
function vi(model::Model, alg::VariationalInference)
    error("Turing.Variational.vi: variational inference algorithm $(typeof(alg)) "
          * "is not implemented")
end
function vi(model::Model, alg::VariationalInference, q::VariationalPosterior)
    error("Turing.Variational.vi: variational inference algorithm $(typeof(alg)) "
          * "is not implemented")
end

# default implementations
function grad!(
    vo,
    alg::VariationalInference{<:ForwardDiffAD},
    q::VariationalPosterior,
    model::Model,
    θ::AbstractVector{<:Real},
    out::DiffResults.MutableDiffResult,
    args...
)
    # TODO: this probably slows down executation quite a bit; exists a better way
    # of doing this?
    f(θ_) = - vo(alg, q, model, θ_, args...)

    chunk_size = getchunksize(typeof(alg))
    # Set chunk size and do ForwardMode.
    chunk = ForwardDiff.Chunk(min(length(θ), chunk_size))
    config = ForwardDiff.GradientConfig(f, θ, chunk)
    ForwardDiff.gradient!(out, f, θ, config)
end

function grad!(
    vo,
    alg::VariationalInference{<:TrackerAD},
    q::VariationalPosterior,
    model::Model,
    θ::AbstractVector{<:Real},
    out::DiffResults.MutableDiffResult,
    args...
)
    θ_tracked = [Tracker.param(θ[i]) for i ∈ eachindex(θ)]
    y = - vo(alg, q, model, θ_tracked, args...)
    Tracker.back!(y, 1.0)

    DiffResults.value!(out, Tracker.data(y))
    DiffResults.gradient!(out, [Tracker.grad(θ_tracked[i]) for i ∈ eachindex(θ_tracked)])
end

"""
    optimize!(vo, alg::VariationalInference{AD}, q::VariationalPosterior, model::Model, θ; optimizer = TruncatedADAGrad())

Iteratively updates parameters by calling `grad!` and using the given `optimizer` to compute
the steps.
"""
function optimize!(
    vo,
    alg::VariationalInference,
    q::VariationalPosterior,
    model::Model,
    θ;
    optimizer = TruncatedADAGrad()
)
    # TODO: should we always assume `samples_per_step` and `max_iters` for all algos?
    alg_name = alg_str(alg)
    samples_per_step = alg.samples_per_step
    max_iters = alg.max_iters

    # number of previous gradients to use to compute `s` in adaGrad
    stepsize_num_prev = 10
    
    num_params = length(q)

    # TODO: really need a better way to warn the user about potentially
    # not using the correct accumulator
    if (optimizer isa TruncatedADAGrad) && (θ ∉ keys(optimizer.acc))
        # this message should only occurr once in the optimization process
        @info "[$alg_name] Should only be seen once: optimizer created for θ" objectid(θ)
    end
    
    diff_result = DiffResults.GradientResult(θ)

    i = 0
    prog = if PROGRESS[]
        ProgressMeter.Progress(max_iters, 1, "[$alg_name] Optimizing...", 0)
    else
        0
    end

    # add criterion? A running mean maybe?
    time_elapsed = @elapsed while (i < max_iters) # & converged
        grad!(vo, alg, q, model, θ, diff_result, samples_per_step)

        # apply update rule
        Δ = DiffResults.gradient(diff_result)
        Δ = apply!(optimizer, θ, Δ)
        @. θ = θ - Δ
        
        Turing.DEBUG && @debug "Step $i" Δ DiffResults.value(diff_result)
        PROGRESS[] && (ProgressMeter.next!(prog))

        i += 1
    end

    return θ
end

# distributions
include("distributions.jl")

# objectives
include("objectives.jl")

# optimisers
include("optimisers.jl")

# VI algorithms
include("advi.jl")

end
