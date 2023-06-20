module ModeEstimation

using ..Turing
using Bijectors
using Random
using SciMLBase: OptimizationFunction, OptimizationProblem, AbstractADType, NoAD

using Setfield
using DynamicPPL
using DynamicPPL: Model, AbstractContext, VarInfo, VarName,
    _getindex, getsym, getfield,  setorder!,
    get_and_set_val!, istrans

import LogDensityProblems
import LogDensityProblemsAD

export  constrained_space,  
        MAP,
        MLE,
        OptimLogDensity,
        OptimizationContext,
        get_parameter_bounds,
        optim_objective, 
        optim_function,
        optim_problem

struct constrained_space{x} end 

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

    function OptimizationContext{C}(context::C) where {C<:AbstractContext}
        leaf = DynamicPPL.leafcontext(context)
        if !(leaf isa Union{DefaultContext,LikelihoodContext})
            throw(ArgumentError("`OptimizationContext` supports only leaf contexts of type `DynamicPPL.DefaultContext` and `DynamicPPL.LikelihoodContext` (given: `$(typeof(leaf)))`"))
        end
        return new{C}(context)
    end
end

OptimizationContext(context::AbstractContext) = OptimizationContext{typeof(context)}(context)

DynamicPPL.NodeTrait(::OptimizationContext) = DynamicPPL.IsParent()
DynamicPPL.childcontext(context::OptimizationContext) = context.context
DynamicPPL.setchildcontext(::OptimizationContext, child) = OptimizationContext(child)

# assume
function DynamicPPL.tilde_assume(ctx::OptimizationContext, dist, vn, vi)
    r = vi[vn, dist]
    lp = if DynamicPPL.leafcontext(ctx) isa DefaultContext
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
    lp = if DynamicPPL.leafcontext(ctx) isa DefaultContext
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



#################################################
# Generic optimisation objective initialisation #
#################################################

function transform!!(f::OptimLogDensity)
    ## Check link status of vi in OptimLogDensity
    linked = DynamicPPL.istrans(f.varinfo)

    ## transform into constrained or unconstrained space depending on current state of vi
    @set! f.varinfo = if !linked
        DynamicPPL.link!!(f.varinfo, f.model)
    else
        DynamicPPL.invlink!!(f.varinfo, f.model)
    end

    return f
end

function transform!!(p::AbstractArray, vi::DynamicPPL.VarInfo, model::DynamicPPL.Model, ::constrained_space{true})
    linked = DynamicPPL.istrans(vi)
    
    !linked && return identity(p)  # TODO: why do we do `identity` here?
    vi = DynamicPPL.unflatten(vi, p)
    vi = DynamicPPL.invlink!!(vi, model)
    p .= vi[:]

    # If linking mutated, we need to link once more.
    linked && DynamicPPL.link!!(vi, model)

    return p
end

function transform!!(p::AbstractArray, vi::DynamicPPL.VarInfo, model::DynamicPPL.Model, ::constrained_space{false})
    linked = DynamicPPL.istrans(vi)
    if linked
        vi = DynamicPPL.invlink!!(vi, model)
    end
    vi = DynamicPPL.unflatten(vi, p)
    vi = DynamicPPL.link!!(vi, model)
    p .= vi[:]

    # If linking mutated, we need to link once more.
    !linked && DynamicPPL.invlink!!(vi, model)

    return p
end

function transform(p::AbstractArray, vi::DynamicPPL.VarInfo, model::DynamicPPL.Model, con::constrained_space)
    return transform!!(copy(p), vi, model, con)
end

abstract type AbstractTransform end

struct ParameterTransform{T<:DynamicPPL.VarInfo,M<:DynamicPPL.Model, S<:constrained_space} <: AbstractTransform
    vi::T
    model::M
    space::S
end

struct Init{T<:DynamicPPL.VarInfo,M<:DynamicPPL.Model, S<:constrained_space} <: AbstractTransform
    vi::T
    model::M
    space::S
end

function (t::AbstractTransform)(p::AbstractArray)
    return transform(p, t.vi, t.model, t.space)
end

function (t::Init)()
    return t.vi[DynamicPPL.SampleFromPrior()]
end 

function get_parameter_bounds(model::DynamicPPL.Model)
    vi = DynamicPPL.VarInfo(model)

    ## Check link status of vi
    linked = DynamicPPL.istrans(vi)
    
    ## transform into unconstrained
    if !linked
        vi = DynamicPPL.link!!(vi, model)
    end

    d = length(vi[:])
    lb = transform(fill(-Inf, d), vi, model, constrained_space{true}())
    ub = transform(fill(Inf, d), vi, model, constrained_space{true}())

    return lb, ub
end

function _optim_objective(model::DynamicPPL.Model, ::MAP, ::constrained_space{false})
    ctx = OptimizationContext(DynamicPPL.DefaultContext())
    obj = OptimLogDensity(model, ctx)

    obj = transform!!(obj)
    init = Init(obj.varinfo, model, constrained_space{false}())
    t = ParameterTransform(obj.varinfo, model, constrained_space{true}())

    return (obj=obj, init = init, transform=t)
end

function _optim_objective(model::DynamicPPL.Model, ::MAP, ::constrained_space{true})
    ctx = OptimizationContext(DynamicPPL.DefaultContext())
    obj = OptimLogDensity(model, ctx)
    
    init = Init(obj.varinfo, model, constrained_space{true}())
    t = ParameterTransform(obj.varinfo, model, constrained_space{true}())
    
    return (obj=obj, init = init, transform=t)
end

function _optim_objective(model::DynamicPPL.Model, ::MLE,  ::constrained_space{false})
    ctx = OptimizationContext(DynamicPPL.LikelihoodContext())
    obj = OptimLogDensity(model, ctx)
    
    obj = transform!!(obj)
    init = Init(obj.varinfo, model, constrained_space{false}())
    t = ParameterTransform(obj.varinfo, model, constrained_space{true}())
    
    return (obj=obj, init = init, transform=t)
end

function _optim_objective(model::DynamicPPL.Model, ::MLE, ::constrained_space{true})
    ctx = OptimizationContext(DynamicPPL.LikelihoodContext())
    obj = OptimLogDensity(model, ctx)
  
    init = Init(obj.varinfo, model, constrained_space{true}())
    t = ParameterTransform(obj.varinfo, model, constrained_space{true}())
    
    return (obj=obj, init = init, transform=t)
end

function optim_objective(model::DynamicPPL.Model, estimator::Union{MLE, MAP}; constrained::Bool=true)
    return _optim_objective(model, estimator, constrained_space{constrained}())
end


function optim_function(
    model::Model,
    estimator::Union{MLE, MAP};
    constrained::Bool=true,
    autoad::Union{Nothing, AbstractADType}=NoAD(),
)
    if autoad === nothing
        Base.depwarn("the use of `autoad=nothing` is deprecated, please use `autoad=SciMLBase.NoAD()`", :optim_function)
    end

    obj, init, t = optim_objective(model, estimator; constrained=constrained)
    
    l(x, _) = obj(x)
    f = if autoad isa AbstractADType && autoad !== NoAD()
        OptimizationFunction(l, autoad)
    else
        OptimizationFunction(
            l;
            grad = (G,x,p) -> obj(nothing, G, x),
        )
    end
    
    return (func=f, init=init, transform = t)
end


function optim_problem(
    model::Model,
    estimator::Union{MAP, MLE};
    constrained::Bool=true,
    init_theta=nothing,
    autoad::Union{Nothing, AbstractADType}=NoAD(),
    kwargs...,
)
    f, init, transform = optim_function(model, estimator; constrained=constrained, autoad=autoad)

    u0 = init_theta === nothing ? init() : init(init_theta)
    prob = OptimizationProblem(f, u0; kwargs...)

    return (; prob, init, transform)
end

end
