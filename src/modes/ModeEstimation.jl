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
    r = vi[vn]
    return r, 0
end

function DynamicPPL.tilde(ctx::OptimizationContext, spl, dist, vn::VarName, inds, vi)
    r = vi[vn]
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
    r = getval(vi, vns)
    return r, 0
end

function DynamicPPL.dot_tilde(ctx::OptimizationContext, sampler, right, left, vn::VarName, _, vi)
    vns, dist = get_vns_and_dist(right, left, vn)
    r = getval(vi, vns)
    return r, loglikelihood(dist, r)
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
    r = getval(vi, vns)
    return loglikelihood(dist, r)
end

function DynamicPPL.dot_tilde(ctx::OptimizationContext, sampler, dists, value, vi)
    return sum(Distributions.logpdf.(dists, value))
end

function getval(
    vi,
    vns::AbstractVector{<:VarName},
)
    r = vi[vns]
    return r
end

function getval(
    vi,
    vns::AbstractArray{<:VarName},
)
    r = reshape(vi[vec(vns)], size(vns))
    return r
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
end

"""
    OptimLogDensity(model::Model, context::AbstractContext)

Create a callable `OptimLogDensity` struct that evaluates a model using the given `context`.
"""
function OptimLogDensity(model::Model, context::AbstractContext)
	init = VarInfo(model)
	DynamicPPL.link!(init, DynamicPPL.SampleFromPrior())
	return OptimLogDensity(model, context, init)
end

"""
    (f::OptimLogDensity)(z)

Evaluate the log joint (with `DefaultContext`) or log likelihood (with `LikelihoodContext`)
at the array `z`.
"""
function (f::OptimLogDensity)(z)
    spl = DynamicPPL.SampleFromPrior()

    varinfo = DynamicPPL.VarInfo(f.vi, spl, z)
    f.model(varinfo, spl, f.context)
    return -DynamicPPL.getlogp(varinfo)
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



function Base.show(io::IO, ::MIME"text/plain", m::ModeResult)
    print(io, "ModeResult with minimized lp of ")
    Printf.@printf(io, "%.2f", m.lp)
    println(io)
    show(io, m.values)
end

function Base.show(io::IO, m::ModeResult)
    show(io, m.values.array)
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
    info = inv(hessian_function(m.f, m.values.array[:, 1]))
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

```julia-repl
@model function f(x)
    m ~ Normal(0, 1)
    x ~ Normal(m, 1)
end

model = f(1.5)
mle = optimize(model, MLE())

# Use a different optimizer
mle = optimize(model, MLE(), NelderMead())
```
"""
function Optim.optimize(model::Model, ::MLE, args...; kwargs...)
    ctx = OptimizationContext(DynamicPPL.LikelihoodContext())
    return optimize(model, OptimLogDensity(model, ctx), args...; kwargs...)
end

"""
    Optim.optimize(model::Model, ::MAP, args...; kwargs...)

Compute a maximum a posterior estimate of the `model`.

# Examples

```julia-repl
@model function f(x)
    m ~ Normal(0, 1)
    x ~ Normal(m, 1)
end

model = f(1.5)
map_est = optimize(model, MAP())

# Use a different optimizer
map_est = optimize(model, MAP(), NelderMead())
```
"""
function Optim.optimize(model::Model, ::MAP, args...; kwargs...)
    ctx = OptimizationContext(DynamicPPL.DefaultContext())
    return optimize(model, OptimLogDensity(model, ctx), args...; kwargs...)
end

"""
    Optim.optimize(model::Model, f::OptimLogDensity, optimizer=Optim.LBFGS(), args...; kwargs...)

Estimate a mode, i.e., compute a MLE or MAP estimate.
"""
function Optim.optimize(model::Model, f::OptimLogDensity, optimizer=Optim.LBFGS(), args...; kwargs...)
    # Do some initialization.
    spl = DynamicPPL.SampleFromPrior()
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
