using ..Turing
using ..Bijectors
using LinearAlgebra

import ..AbstractMCMC: AbstractSampler
import ..DynamicPPL
import ..DynamicPPL: Model, AbstractContext, VarInfo, AbstractContext, VarName,
    _getindex, getsym, getfield, settrans!,  setorder!,
    get_and_set_val!, istrans, tilde, dot_tilde
import Optim
import Optim: optimize
import NamedArrays
import ..ForwardDiff
import StatsBase
import Printf

export MAP, MLE, optimize

struct MLE end
struct MAP end

"""
    OptimizationContext{C<:AbstractContext} <: AbstractContext

The `OptimizationContext` transforms variables to their constrained space, but
does not use the density with respect to the transformation. This context is
intended to allow an optimizer to sample in R^n freely.
"""
struct OptimizationContext{C<:AbstractContext} <: AbstractContext
	context::C
end

# assume
function DynamicPPL.tilde(ctx::OptimizationContext{<:LikelihoodContext}, spl, dist, vn::VarName, inds, vi)
    if haskey(vi, vn)
        # Always overwrite the parameters with new ones for `SampleFromUniform`.
        if spl isa SampleFromUniform || is_flagged(vi, vn, "del")
            unset_flag!(vi, vn, "del")
            r = init(dist, spl)
            vi[vn] = vectorize(dist, r)
            settrans!(vi, false, vn)
            setorder!(vi, vn, get_num_produce(vi))
        else
            r = vi[vn]
        end
    else
        r = init(dist, spl)
        push!(vi, vn, r, dist, spl)
        settrans!(vi, false, vn)
    end
	return r, 0
end

function DynamicPPL.tilde(ctx::OptimizationContext, spl, dist, vn::VarName, inds, vi)
    if haskey(vi, vn)
        # Always overwrite the parameters with new ones for `SampleFromUniform`.
        if spl isa SampleFromUniform || is_flagged(vi, vn, "del")
            unset_flag!(vi, vn, "del")
            r = init(dist, spl)
            vi[vn] = vectorize(dist, r)
            settrans!(vi, false, vn)
            setorder!(vi, vn, get_num_produce(vi))
        else
            r = vi[vn]
        end
    else
        r = init(dist, spl)
        push!(vi, vn, r, dist, spl)
        settrans!(vi, false, vn)
    end
    return r, Distributions.logpdf(dist, r)
end

# observe
function DynamicPPL.tilde(ctx::OptimizationContext{<:PriorContext}, sampler, right, left, vi)
    return 0
end

function DynamicPPL.tilde(ctx::OptimizationContext, sampler, dist, value, vi)
    return Distributions.logpdf(dist, value)
end

# dot assume
function DynamicPPL.dot_tilde(ctx::OptimizationContext{<:LikelihoodContext}, sampler, right, left, vn::VarName, _, vi)
    vns, dist = get_vns_and_dist(right, left, vn)
    r = get_and_set_val!(vi, vns, dist, sampler)
    return r, 0
end

function DynamicPPL.dot_tilde(ctx::OptimizationContext, sampler, right, left, vn::VarName, _, vi)
    vns, dist = get_vns_and_dist(right, left, vn)
    r = get_and_set_val!(vi, vns, dist, sampler)
    lp = sum(logpdf(dist, r))
    return r, lp
end

# dot observe
function DynamicPPL.dot_tilde(ctx::OptimizationContext{<:PriorContext}, sampler, right, left, vn, _, vi)
    return 0
end

function DynamicPPL.dot_tilde(ctx::OptimizationContext{<:PriorContext}, sampler, right, left, vi)
    return 0
end

function DynamicPPL.dot_tilde(ctx::OptimizationContext, sampler, right, left, vn, _, vi)
    vns, dist = get_vns_and_dist(right, left, vn)
    r = get_and_set_val!(vi, vns, dist, sampler)
    lp = sum(logpdf(dist, r))
    return lp
end

function DynamicPPL.dot_tilde(ctx::OptimizationContext, sampler, dists, value, vi)
    return sum(Distributions.logpdf.(dists, value))
end

"""
    OptimLogDensity{M<:Model,C<:Context,V<:VarInfo}

A struct that stores the log density function of a `DynamicPPL` model.
"""
struct OptimLogDensity{M<:Model,C<:AbstractContext,V<:VarInfo}
    "A `DynamicPPL.Model` constructed either with the `@model` macro or manually."
    model::M
    "A `DynamicPPL.AbstractContext` used to evaluate the model. `LikelihoodContext` or `DefaultContext` are typical for MAP/MLE."
    context::C
    "A `DynamicPPL.VarInfo` struct that will be used to update model parameters."
    vi::V
    "Whether the to evaluate the model with transformed variables."
    linked::Bool
end

"""
    OptimLogDensity(model::Model, context::AbstractContext)

Create a callable `OptimLogDensity` struct that evaluates a model using the given `context`.
"""
function OptimLogDensity(model::Model, context::AbstractContext, linked::Bool)
	init = VarInfo(model, context)
	linked && DynamicPPL.link!(init, DynamicPPL.SampleFromPrior())
	return OptimLogDensity(model, context, init, linked)
end

"""
    (f::OptimLogDensity)(z; unlinked::Bool = false)

Evaluate the log joint (with `DefaultContext`) or log likelihood (with `LikelihoodContext`)
at the array `z`. If `unlinked=true`, no change of variables will occur.
"""
function (f::OptimLogDensity)(z; linked::Bool = true)
    spl = DynamicPPL.SampleFromPrior()

    varinfo = DynamicPPL.VarInfo(f.vi, spl, z)
    f.model(varinfo, spl, f.context)
    return -DynamicPPL.getlogp(varinfo)
end

"""
    unlink(f::OptimLogDensity)

Generate an unlinked (with no variable transformions) version of an existing `OptimLogDensity`.
"""
function Bijectors.unlink(f::OptimLogDensity)
    init = VarInfo(f.model, f.context)
    return OptimLogDensity(f.model, f.context, init, false)
end

"""
    link(f::OptimLogDensity)

Generate an linked (with variable transformions) version of an existing `OptimLogDensity`.
"""
function Bijectors.link(f::OptimLogDensity)
    init = VarInfo(f.model, f.context)
    DynamicPPL.link!(init, DynamicPPL.SampleFromPrior())
    return OptimLogDensity(f.model, f.context, init, false)
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
    O<:Optim.MultivariateOptimizationResults,
    M<:OptimLogDensity
} <: StatsBase.StatisticalModel
    "A vector with the resulting point estimates."
    values :: V
    "The stored Optim.jl results."
    optim_result :: O
    "The final log likelihood or log joint, depending on whether `MAP` or `MLE` was run."
    lp :: Float64
    "The evaluation function used to calculate the output."
    f :: M
end
#############################
# Various StatsBase methods #
#############################

function Base.show(io::IO, m::ModeResult)
    println(io, Printf.@sprintf("ModeResult with minimized lp of %.2f", m.lp), "\n")
    show(io, m.values)
end

function StatsBase.coeftable(m::ModeResult)
    # Get columns for coeftable.
    terms = StatsBase.coefnames(m)
    estimates = m.values.array[:,1]
    stderrors = StatsBase.stderror(m)
    tstats = estimates ./ stderrors

    StatsBase.CoefTable([estimates, stderrors, tstats], ["estimate", "stderror", "tstat"], terms)
end

function StatsBase.informationmatrix(m::ModeResult; hessian_function=ForwardDiff.hessian, kwargs...)
    # Calculate Hessian and information matrix.
    varnames = StatsBase.coefnames(m)
    f = unlink(m.f)
    info = inv(hessian_function(x -> f(x), m.values.array[:, 1]))
    return NamedArrays.NamedArray(info, (varnames, varnames))
end

StatsBase.coef(m::ModeResult) = m.values
StatsBase.coefnames(m::ModeResult) = names(m.values)[1]
StatsBase.params(m::ModeResult) = StatsBase.coefnames(m)
StatsBase.vcov(m::ModeResult) = StatsBase.informationmatrix(m)
StatsBase.loglikelihood(m::ModeResult) = m.lp

####################
# Optim.jl methods #
####################

"""
    Optim.optimize(model::Model, ::MLE, args...; kwargs...)

Compute a maximum likelihood estimate of the `model`.

# Examples

```julia
@model function f(x)
    m ~ Normal(0, 1)
    x ~ Normal(m, 1)
end

model = f(1.5)
mle = optimize(model, MLE())

# Use a different optimizer
mle = optimize(model, MLE(), NelderMeade())
```
"""
function Optim.optimize(model::Model, ::MLE, args...; kwargs...)
    ctx = OptimizationContext(DynamicPPL.LikelihoodContext())
    return optimize(model, OptimLogDensity(model, ctx, true), args...; kwargs...)
end

"""
    Optim.optimize(model::Model, ::MAP, args...; kwargs...)

Compute a maximum a posterior estimate of the `model`.

# Examples

```julia
@model function f(x)
    m ~ Normal(0, 1)
    x ~ Normal(m, 1)
end

model = f(1.5)
map_est = optimize(model, MAP())

# Use a different optimizer
map_est = optimize(model, MAP(), NelderMeade())
```
"""
function Optim.optimize(model::Model, ::MAP, args...; kwargs...)
    ctx = OptimizationContext(DynamicPPL.DefaultContext())
    return optimize(model, OptimLogDensity(model, ctx, true), args...; kwargs...)
end

"""
    Optim.optimize(model::Model, f::OptimLogDensity, optimizer=Optim.LBFGS(), args...; kwargs...)

Estimate a mode, i.e., compute a MLE or MAP estimate.
"""
function Optim.optimize(model::Model, f::OptimLogDensity, optimizer=Optim.LBFGS(), args...; kwargs...)
    # Do some initialization.
    spl = DynamicPPL.SampleFromPrior()
    init_params = model(f.vi, spl)
    init_vals = f.vi[spl]

    # Optimize!
    M = Optim.optimize(f, init_vals, optimizer, args...; kwargs...)

    # Get the VarInfo at the MLE/MAP point, and run the model to ensure 
    # correct dimensionality.
    f.vi[spl] = M.minimizer
    invlink!(f.vi, spl)
    vals = f.vi[spl]

    # Make one transition to get the parameter names.
    ts = [Turing.Inference.Transition(DynamicPPL.tonamedtuple(f.vi), DynamicPPL.getlogp(f.vi))]
    varnames, _ = Turing.Inference._params_to_array(ts)

    # Store the parameters and their names in an array.
    vmat = NamedArrays.NamedArray(vals, varnames)

    return ModeResult(vmat, M, -M.minimum, f)
end
