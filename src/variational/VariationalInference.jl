module Variational

using ..Core, ..Utilities
using DocStringExtensions: TYPEDEF, TYPEDFIELDS
using Distributions, Bijectors, DynamicPPL
using LinearAlgebra
using ..Turing: PROGRESS, Turing
using DynamicPPL: Model, SampleFromPrior, SampleFromUniform
using Random: AbstractRNG

using ForwardDiff
using Tracker

import ..Core: getchunksize, getADbackend

import AbstractMCMC
import ProgressLogging

using Requires
function __init__()
    @require Flux="587475ba-b771-5e3f-ad9e-33799f191a9c" begin
        apply!(o, x, Δ) = Flux.Optimise.apply!(o, x, Δ)
        Flux.Optimise.apply!(o::TruncatedADAGrad, x, Δ) = apply!(o, x, Δ)
        Flux.Optimise.apply!(o::DecayedADAGrad, x, Δ) = apply!(o, x, Δ)
    end
    @require Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f" begin
        function Variational.grad!(
            vo,
            alg::VariationalInference{<:Turing.ZygoteAD},
            q,
            model,
            θ::AbstractVector{<:Real},
            out::DiffResults.MutableDiffResult,
            args...
        )
            f(θ) = if (q isa VariationalPosterior)
                - vo(alg, update(q, θ), model, args...)
            else
                - vo(alg, q(θ), model, args...)
            end
            y, back = Tracker.pullback(f, θ)
            dy = back(1.0)
            DiffResults.value!(out, y)
            DiffResults.gradient!(out, dy)
            return out
        end
    end
    @require ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267" begin
        function Variational.grad!(
            vo,
            alg::VariationalInference{<:Turing.ReverseDiffAD{false}},
            q,
            model,
            θ::AbstractVector{<:Real},
            out::DiffResults.MutableDiffResult,
            args...
        )
            f(θ) = if (q isa VariationalPosterior)
                - vo(alg, update(q, θ), model, args...)
            else
                - vo(alg, q(θ), model, args...)
            end
            tp = Turing.Core.tape(f, θ)
            ReverseDiff.gradient!(out, tp, θ)
            return out
        end
        @require Memoization = "6fafb56a-5788-4b4e-91ca-c0cea6611c73" begin
            function Variational.grad!(
                vo,
                alg::VariationalInference{<:Turing.ReverseDiffAD{true}},
                q,
                model,
                θ::AbstractVector{<:Real},
                out::DiffResults.MutableDiffResult,
                args...
            )
                f(θ) = if (q isa VariationalPosterior)
                    - vo(alg, update(q, θ), model, args...)
                else
                    - vo(alg, q(θ), model, args...)
                end
                ctp = Turing.Core.memoized_tape(f, θ)
                ReverseDiff.gradient!(out, ctp, θ)
                return out
            end
        end
    end
end

export
    vi,
    ADVI,
    ELBO,
    elbo,
    TruncatedADAGrad,
    DecayedADAGrad

abstract type VariationalInference{AD} end

getchunksize(::Type{<:VariationalInference{AD}}) where AD = getchunksize(AD)
getADbackend(::VariationalInference{AD}) where AD = AD()

abstract type VariationalObjective end

const VariationalPosterior = Distribution{Multivariate, Continuous}


"""
    grad!(vo, alg::VariationalInference, q, model::Model, θ, out, args...)

Computes the gradients used in `optimize!`. Default implementation is provided for
`VariationalInference{AD}` where `AD` is either `ForwardDiffAD` or `TrackerAD`.
This implicitly also gives a default implementation of `optimize!`.

Variance reduction techniques, e.g. control variates, should be implemented in this function.
"""
function grad! end

"""
    vi(model, alg::VariationalInference)
    vi(model, alg::VariationalInference, q::VariationalPosterior)
    vi(model, alg::VariationalInference, getq::Function, θ::AbstractArray)

Constructs the variational posterior from the `model` and performs the optimization
following the configuration of the given `VariationalInference` instance.

# Arguments
- `model`: `Turing.Model` or `Function` z ↦ log p(x, z) where `x` denotes the observations
- `alg`: the VI algorithm used
- `q`: a `VariationalPosterior` for which it is assumed a specialized implementation of the variational objective used exists.
- `getq`: function taking parameters `θ` as input and returns a `VariationalPosterior`
- `θ`: only required if `getq` is used, in which case it is the initial parameters for the variational posterior
"""
function vi end

# default implementations
function grad!(
    vo,
    alg::VariationalInference{<:ForwardDiffAD},
    q,
    model,
    θ::AbstractVector{<:Real},
    out::DiffResults.MutableDiffResult,
    args...
)
    f(θ_) = if (q isa VariationalPosterior)
        - vo(alg, update(q, θ_), model, args...)
    else
        - vo(alg, q(θ_), model, args...)
    end

    chunk_size = getchunksize(typeof(alg))
    # Set chunk size and do ForwardMode.
    chunk = ForwardDiff.Chunk(min(length(θ), chunk_size))
    config = ForwardDiff.GradientConfig(f, θ, chunk)
    ForwardDiff.gradient!(out, f, θ, config)
end

function grad!(
    vo,
    alg::VariationalInference{<:TrackerAD},
    q,
    model,
    θ::AbstractVector{<:Real},
    out::DiffResults.MutableDiffResult,
    args...
)
    θ_tracked = Tracker.param(θ)
    y = if (q isa VariationalPosterior)
        - vo(alg, update(q, θ_tracked), model, args...)
    else
        - vo(alg, q(θ_tracked), model, args...)
    end
    Tracker.back!(y, 1.0)

    DiffResults.value!(out, Tracker.data(y))
    DiffResults.gradient!(out, Tracker.grad(θ_tracked))
end

"""
    optimize!(vo, alg::VariationalInference{AD}, q::VariationalPosterior, model, θ; optimizer = TruncatedADAGrad())

Iteratively updates parameters by calling `grad!` and using the given `optimizer` to compute
the steps.
"""
function optimize!(
    vo,
    alg::VariationalInference,
    q,
    model,
    θ::AbstractVector{<:Real};
    optimizer = TruncatedADAGrad(),
    progress = Turing.PROGRESS[],
    progressname = "[$(DynamicPPL.alg_str(alg))] Optimizing..."
)
    # TODO: should we always assume `samples_per_step` and `max_iters` for all algos?
    samples_per_step = alg.samples_per_step
    max_iters = alg.max_iters

    # TODO: really need a better way to warn the user about potentially
    # not using the correct accumulator
    if optimizer isa TruncatedADAGrad && θ ∉ keys(optimizer.acc)
        # this message should only occurr once in the optimization process
        @info "[$(DynamicPPL.alg_str(alg))] Should only be seen once: optimizer created for θ" objectid(θ)
    end

    diff_result = DiffResults.GradientResult(θ)

    # Create the progress bar.
    AbstractMCMC.@ifwithprogresslogger progress name=progressname begin
        # add criterion? A running mean maybe?
        for i in 1:max_iters
            grad!(vo, alg, q, model, θ, diff_result, samples_per_step)

            # apply update rule
            Δ = DiffResults.gradient(diff_result)
            Δ = apply!(optimizer, θ, Δ)
            @. θ = θ - Δ

            Turing.DEBUG && @debug "Step $i" Δ DiffResults.value(diff_result)

            # Update the progress bar.
            progress && ProgressLogging.@logprogress i/max_iters
        end
    end

    return θ
end

"""
    make_logjoint(model::Model; weight = 1.0)

Constructs the logjoint as a function of latent variables, i.e. the map z → p(x ∣ z) p(z).

The weight used to scale the likelihood, e.g. when doing stochastic gradient descent one needs to
use `DynamicPPL.MiniBatch` context to run the `Model` with a weight `num_total_obs / batch_size`.

## Notes
- For sake of efficiency, the returned function is closes over an instance of `VarInfo`. This means that you *might* run into some weird behaviour if you call this method sequentially using different types; if that's the case, just generate a new one for each type using `make_logjoint`.
"""
function make_logjoint(model::Model; weight = 1.0)
    # setup
    ctx = DynamicPPL.MiniBatchContext(
        DynamicPPL.DefaultContext(),
        weight
    )
    varinfo_init = Turing.VarInfo(model, ctx)

    function logπ(z)
        varinfo = VarInfo(varinfo_init, SampleFromUniform(), z)
        model(varinfo)

        return getlogp(varinfo)
    end

    return logπ
end

function logjoint(model::Model, varinfo, z)
    varinfo = VarInfo(varinfo, SampleFromUniform(), z)
    model(varinfo)

    return getlogp(varinfo)
end


# objectives
include("objectives.jl")

# optimisers
include("optimisers.jl")

# VI algorithms
include("advi.jl")

end
