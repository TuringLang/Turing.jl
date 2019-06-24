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
    vo, vi::VariationalInference{AD},
    q::VariationalPosterior,
    model::Model,
    θ,
    out,
    args...
) where AD
    error("Turing.Variational.grad!: unmanaged variational inference algorithm: "
          "$(typeof(alg))")
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
    alg::VariationalInference{AD},
    q::VariationalPosterior,
    model::Model,
    θ::AbstractVector{T},
    out::DiffResults.MutableDiffResult,
    args...
) where {T <: Real, AD <: ForwardDiffAD}
    # TODO: this probably slows down executation quite a bit; exists a better way
    # of doing this?
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
function grad!(
    vo,
    alg::VariationalInference{AD},
    q::VariationalPosterior,
    model::Model,
    θ::AbstractVector{T},
    out::DiffResults.MutableDiffResult,
    args...
) where {T <: Real, AD <: TrackerAD}
    θ_tracked = Tracker.param(θ)
    y = - vo(alg, q, model, θ_tracked, args...)
    Tracker.back!(y, 1.0)

    DiffResults.value!(out, Tracker.data(y))
    DiffResults.gradient!(out, Tracker.grad(θ_tracked))
end

"""
    optimize!(vo, alg::VariationalInference{AD}, q::VariationalPosterior, model::Model, θ; optimizer = ADAGrad())

Iteratively updates parameters by calling `grad!` and using the given `optimizer` to compute
the steps.
"""
function optimize!(
    vo,
    alg::VariationalInference{AD},
    q::VariationalPosterior,
    model::Model,
    θ;
    optimizer = ADAGrad()
) where AD
    # TODO: this default constructor for `optimizer` is probably dangerous because of
    # the `acc` field being an `IdDict`!!!

    # TODO: should we always assume `samples_per_step` and `max_iters` for all algos?
    alg_name = alg_str(alg)
    samples_per_step = alg.samples_per_step
    max_iters = alg.max_iters

    # number of previous gradients to use to compute `s` in adaGrad
    stepsize_num_prev = 10
    
    # setup
    # var_info = Turing.VarInfo()
    # model(var_info, Turing.SampleFromUniform())
    # num_params = size(var_info.vals, 1)
    num_params = length(q)

    # # buffer
    # θ = zeros(2 * num_params)

    # HACK: re-use previous gradient `acc` if equal in value
    # Can cause issues if two entries have idenitical values
    if θ ∉ keys(optimizer.acc)
        vs = [v for v ∈ keys(optimizer.acc)]
        idx = findfirst(w -> vcat(q.μ, q.ω) == w, vs)
        if idx != nothing
            @info "[$alg_name] Re-using previous optimizer accumulator"
            θ .= vs[idx]
        end
    else
        @info "[$alg_name] Already present in optimizer acc"
    end
    
    diff_result = DiffResults.GradientResult(θ)

    # TODO: in (Blei et al, 2015) TRUNCATED ADAGrad is suggested;
    # this is not available in Flux.Optimise
    # Maybe consider contributing a truncated ADAGrad to Flux.Optimise

    i = 0
    prog = if PROGRESS[]
        ProgressMeter.Progress(max_iters, 1, "[$alg_name] Optimizing...", 0)
    else
        0
    end

    # add criterion? A running mean maybe?
    time_elapsed = @elapsed while (i < max_iters) # & converged
        # TODO: separate into a `grad(...)` call;
        # need to manually provide `diff_result` buffers
        # ForwardDiff.gradient!(diff_result, f, x)
        grad!(vo, alg, q, model, θ, diff_result, samples_per_step)

        # apply update rule
        Δ = DiffResults.gradient(diff_result)
        Δ = Optimise.apply!(optimizer, θ, Δ)
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

# VI algorithms
include("advi.jl")

end
