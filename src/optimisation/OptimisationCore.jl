"""Module for the doing MLE and MAP optimisation in Turing.jl.

This module provides the core tools for optimisation. It is used by the user facing modules
and extensions for Optim.jl and Optimisation.jl.
"""
module OptimisationCore

using ..Turing
using NamedArrays
using DynamicPPL
using DynamicPPL: Model, AbstractContext, VarInfo, get_and_set_val!, istrans
import LogDensityProblems
import LogDensityProblemsAD
using StatsBase
using Accessors: Accessors
using Printf
using ForwardDiff
using StatsAPI
using Statistics
using Random

export ModeResult, OptimizationContext, OptimLogDensity, MLE, MAP, ModeEstimator


abstract type ModeEstimator end
struct MLE <: ModeEstimator end
struct MAP <: ModeEstimator end


"""
    OptimizationContext{C<:AbstractContext} <: AbstractContext

The `OptimizationContext` transforms variables to their constrained space, but
does not use the density with respect to the transformation. This context is
intended to allow an optimizer to sample in R^n freely.
"""
struct OptimizationContext{C<:AbstractContext} <: AbstractContext
    context::C

    function OptimizationContext{C}(context::C) where {C<:AbstractContext}
        if !(context isa Union{DefaultContext,LikelihoodContext})
            throw(ArgumentError("`OptimizationContext` supports only leaf contexts of type `DynamicPPL.DefaultContext` and `DynamicPPL.LikelihoodContext` (given: `$(typeof(context)))`"))
        end
        return new{C}(context)
    end
end

OptimizationContext(context::AbstractContext) = OptimizationContext{typeof(context)}(context)

DynamicPPL.NodeTrait(::OptimizationContext) = DynamicPPL.IsLeaf()

# assume
function DynamicPPL.tilde_assume(ctx::OptimizationContext, dist, vn, vi)
    r = vi[vn, dist]
    lp = if ctx.context isa DefaultContext
        # MAP
        Distributions.logpdf(dist, r)
    else
        # MLE
        0
    end
    return r, lp, vi
end

# dot assume
_loglikelihood(dist::Distribution, x) = loglikelihood(dist, x)
_loglikelihood(dists::AbstractArray{<:Distribution}, x) = loglikelihood(arraydist(dists), x)
function DynamicPPL.dot_tilde_assume(ctx::OptimizationContext, right, left, vns, vi)
    # Values should be set and we're using `SampleFromPrior`, hence the `rng` argument shouldn't
    # affect anything.
    # TODO: Stop using `get_and_set_val!`.
    r = DynamicPPL.get_and_set_val!(Random.default_rng(), vi, vns, right, SampleFromPrior())
    lp = if ctx.context isa DefaultContext
        # MAP
        _loglikelihood(right, r)
    else
        # MLE
        0
    end
    return r, lp, vi
end

"""
    OptimLogDensity{M<:Model,C<:Context,V<:VarInfo}

A struct that stores the negative log density function of a `DynamicPPL` model.
"""
const OptimLogDensity{M<:Model,C<:OptimizationContext,V<:VarInfo} = Turing.LogDensityFunction{V,M,C}

"""
    OptimLogDensity(model::Model, context::OptimizationContext)

Create a callable `OptimLogDensity` struct that evaluates a model using the given `context`.
"""
function OptimLogDensity(model::Model, context::OptimizationContext)
    init = VarInfo(model)
    return Turing.LogDensityFunction(init, model, context)
end

"""
    LogDensityProblems.logdensity(f::OptimLogDensity, z)

Evaluate the negative log joint (with `DefaultContext`) or log likelihood (with `LikelihoodContext`)
at the array `z`.
"""
function (f::OptimLogDensity)(z::AbstractVector)
    varinfo = DynamicPPL.unflatten(f.varinfo, z)
    return -getlogp(last(DynamicPPL.evaluate!!(f.model, varinfo, f.context)))
end

# NOTE: This seems a bit weird IMO since this is the _negative_ log-likelihood.
LogDensityProblems.logdensity(f::OptimLogDensity, z::AbstractVector) = f(z)

# NOTE: The format of this function is dictated by Optim. The first argument sets whether to
# compute the function value, the second whether to compute the gradient (and stores the
# gradient). The last one is the actual argument of the objective function.
function (f::OptimLogDensity)(F, G, z)
    if G !== nothing
        # Calculate negative log joint and its gradient.
        # TODO: Make OptimLogDensity already an LogDensityProblems.ADgradient? Allow to specify AD?
        ℓ = LogDensityProblemsAD.ADgradient(f)
        neglogp, ∇neglogp = LogDensityProblems.logdensity_and_gradient(ℓ, z)

        # Save the gradient to the pre-allocated array.
        copyto!(G, ∇neglogp)

        # If F is something, the negative log joint is requested as well.
        # We have already computed it as a by-product above and hence return it directly.
        if F !== nothing
            return neglogp
        end
    end

    # Only negative log joint requested but no gradient.
    if F !== nothing
        return LogDensityProblems.logdensity(f, z)
    end

    return nothing
end



"""
    ModeResult{
        V<:NamedArrays.NamedArray,
        M<:NamedArrays.NamedArray,
        O<:Optim.MultivariateOptimizationResults,
        S<:NamedArrays.NamedArray
    }

A wrapper struct to store various results from a MAP or MLE estimation.
"""
struct ModeResult{
    V<:NamedArrays.NamedArray,
    O<:Any,
    M<:OptimLogDensity
} <: StatsBase.StatisticalModel
    "A vector with the resulting point estimates."
    values::V
    "The stored Optim.jl results."
    optim_result::O
    "The final log likelihood or log joint, depending on whether `MAP` or `MLE` was run."
    lp::Float64
    "The evaluation function used to calculate the output."
    f::M
end

#############################
# Various StatsBase methods #
#############################

function Base.show(io::IO, ::MIME"text/plain", m::ModeResult)
    print(io, "ModeResult with maximized lp of ")
    Printf.@printf(io, "%.2f", m.lp)
    println(io)
    show(io, m.values)
end

function Base.show(io::IO, m::ModeResult)
    show(io, m.values.array)
end

function StatsBase.coeftable(m::ModeResult; level::Real=0.95)
    # Get columns for coeftable.
    terms = string.(StatsBase.coefnames(m))
    estimates = m.values.array[:, 1]
    stderrors = StatsBase.stderror(m)
    zscore = estimates ./ stderrors
    p = map(z -> StatsAPI.pvalue(Distributions.Normal(), z; tail=:both), zscore)

    # Confidence interval (CI)
    q = Statistics.quantile(Distributions.Normal(), (1 + level) / 2)
    ci_low = estimates .- q .* stderrors
    ci_high = estimates .+ q .* stderrors

    level_ = 100 * level
    level_percentage = isinteger(level_) ? Int(level_) : level_

    StatsBase.CoefTable(
        [estimates, stderrors, zscore, p, ci_low, ci_high],
        ["Coef.", "Std. Error", "z", "Pr(>|z|)", "Lower $(level_percentage)%", "Upper $(level_percentage)%"],
        terms)
end

function StatsBase.informationmatrix(m::ModeResult; hessian_function=ForwardDiff.hessian, kwargs...)
    # Calculate Hessian and information matrix.

    # Convert the values to their unconstrained states to make sure the
    # Hessian is computed with respect to the untransformed parameters.
    linked = DynamicPPL.istrans(m.f.varinfo)
    if linked
        m = Accessors.@set m.f.varinfo = DynamicPPL.invlink!!(m.f.varinfo, m.f.model)
    end

    # Calculate the Hessian, which is the information matrix because the negative of the log likelihood was optimized
    varnames = StatsBase.coefnames(m)
    info = hessian_function(m.f, m.values.array[:, 1])

    # Link it back if we invlinked it.
    if linked
        m = Accessors.@set m.f.varinfo = DynamicPPL.link!!(m.f.varinfo, m.f.model)
    end

    return NamedArrays.NamedArray(info, (varnames, varnames))
end

StatsBase.coef(m::ModeResult) = m.values
StatsBase.coefnames(m::ModeResult) = names(m.values)[1]
StatsBase.params(m::ModeResult) = StatsBase.coefnames(m)
StatsBase.vcov(m::ModeResult) = inv(StatsBase.informationmatrix(m))
StatsBase.loglikelihood(m::ModeResult) = m.lp

end
