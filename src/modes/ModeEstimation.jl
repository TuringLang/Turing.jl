using ..Turing
using ..Bijectors
using LinearAlgebra

import ..DynamicPPL
import ..DynamicPPL: Model, AbstractContext, VarInfo
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
    OptimLogDensity{M<:Model,C<:Context,V<:VarInfo}

A struct that stores the log density function of a `DynamicPPL` model.
"""
struct OptimLogDensity{M<:Model,C<:AbstractContext,V<:VarInfo}
    "A `DynamicPPL.Model` constructed either with the `@model` macro or manually."
    model::M
    "A `DynamicPPL.AbstractContext` used to evaluate the model. `LikelihoodContext` or `DefaultContext` are typical for MAP/MLE."
    context::C
    "A `DynamicPPL.VarInfo` struct that will be used to update model parameters."
	init::V
end

function OptimLogDensity(model::Model, context::AbstractContext)
	init = VarInfo(model)
	DynamicPPL.link!(init, SampleFromPrior())
	return OptimLogDensity(model, context, init)
end

function (f::OptimLogDensity)(z; unlinked::Bool = false)
    spl = DynamicPPL.SampleFromPrior()

    varinfo = DynamicPPL.VarInfo(f.init, spl, z)

    unlinked && DynamicPPL.invlink!(f.init, spl)
    f.model(varinfo, spl, f.context)
    unlinked && DynamicPPL.link!(f.init, spl)

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
    info = hessian_function(x -> m.f(x, unlinked=true), m.values.array[:, 1])
    info = inv(info)
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

function Optim.optimize(model::Model, ::MLE, args...; kwargs...)
    return optimize(model, OptimLogDensity(model, DynamicPPL.LikelihoodContext()), args...; kwargs...)
end

function Optim.optimize(model::Model, ::MAP, args...; kwargs...)
    return optimize(model, OptimLogDensity(model, DynamicPPL.DefaultContext()), args...; kwargs...)
end

function Optim.optimize(model::Model, f::OptimLogDensity, optimizer=Optim.BFGS(), args...; kwargs...)
    # Do some initialization.
    b = bijector(model)
    binv = inv(b)

    spl = DynamicPPL.SampleFromPrior()
    vi = DynamicPPL.VarInfo(model)
    init_params = model(vi, spl)
    init_vals = vi[spl]

    # Optimize!
    M = Optim.optimize(f, init_vals, args...; kwargs...)

    # Retrieve the estimated values.
    vals = binv(M.minimizer)

    # Get the VarInfo at the MLE/MAP point, and run the model to ensure 
    # correct dimensionality.
    vi[spl] = vals
    model(vi) # XXX: Is this a necessary step?

    # Make one transition to get the parameter names.
    ts = [Turing.Inference.Transition(DynamicPPL.tonamedtuple(vi), DynamicPPL.getlogp(vi))]
    varnames, _ = Turing.Inference._params_to_array(ts)

    # Store the parameters and their names in an array.
    vmat = NamedArrays.NamedArray(vals, varnames)

    return ModeResult(vmat, M, -M.minimum, f)
end