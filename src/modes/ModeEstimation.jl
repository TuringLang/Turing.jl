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
end

DynamicPPL.NodeTrait(::OptimizationContext) = DynamicPPL.IsParent()
DynamicPPL.childcontext(context::OptimizationContext) = context.context
DynamicPPL.setchildcontext(::OptimizationContext, child) = OptimizationContext(child)

# assume
function DynamicPPL.tilde_assume(ctx::OptimizationContext{<:LikelihoodContext}, dist, vn, vi)
    r = vi[vn, dist]
    return r, 0, vi
end

function DynamicPPL.tilde_assume(ctx::OptimizationContext, dist, vn, vi)
    r = vi[vn, dist]
    return r, Distributions.logpdf(dist, r), vi
end

# dot assume
function DynamicPPL.dot_tilde_assume(ctx::OptimizationContext{<:LikelihoodContext}, right, left, vns, vi)
    # Values should be set and we're using `SampleFromPrior`, hence the `rng` argument shouldn't
    # affect anything.
    # TODO: Stop using `get_and_set_val!`.
    r = DynamicPPL.get_and_set_val!(Random.default_rng(), vi, vns, right, SampleFromPrior())
    return r, 0, vi
end

function DynamicPPL.dot_tilde_assume(ctx::OptimizationContext, right, left, vns, vi)
    # Values should be set and we're using `SampleFromPrior`, hence the `rng` argument shouldn't
    # affect anything.
    # TODO: Stop using `get_and_set_val!`.
    r = DynamicPPL.get_and_set_val!(Random.default_rng(), vi, vns, right, SampleFromPrior())
    return r, loglikelihood(right, r), vi
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
    (f::OptimLogDensity)(z)

Evaluate the negative log joint (with `DefaultContext`) or log likelihood (with `LikelihoodContext`)
at the array `z`.
"""
function (f::OptimLogDensity)(z::AbstractVector)
    varinfo = DynamicPPL.unflatten(f.varinfo, z)
    return -getlogp(last(DynamicPPL.evaluate!!(f.model, varinfo, f.context)))
end

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
        return f(z)
    end

    return nothing
end



#################################################
# Generic optimisation objective initialisation #
#################################################

function transform!!(f::OptimLogDensity)
    # TODO: Do we really need this?
    spl = DynamicPPL.SampleFromPrior()

    ## Check link status of vi in OptimLogDensity
    linked = DynamicPPL.islinked(f.varinfo, spl)

    ## transform into constrained or unconstrained space depending on current state of vi
    @set! f.varinfo = if !linked
        DynamicPPL.link!!(f.varinfo, spl, f.model)
    else
        DynamicPPL.invlink!!(f.varinfo, spl, f.model)
    end

    return f
end

function transform!!(p::AbstractArray, vi::DynamicPPL.VarInfo, model::DynamicPPL.Model, ::constrained_space{true})
    spl = DynamicPPL.SampleFromPrior()

    linked = DynamicPPL.islinked(vi, spl)
    
    !linked && return identity(p)  # TODO: why do we do `identity` here?
    vi = DynamicPPL.setindex!!(vi, p, spl)
    vi = DynamicPPL.invlink!!(vi, spl, model)
    p .= vi[spl]

    # If linking mutated, we need to link once more.
    linked && DynamicPPL.link!!(vi, spl, model)

    return p
end

function transform!!(p::AbstractArray, vi::DynamicPPL.VarInfo, model::DynamicPPL.Model, ::constrained_space{false})
    spl = DynamicPPL.SampleFromPrior()

    linked = DynamicPPL.islinked(vi, spl)
    if linked
        vi = DynamicPPL.invlink!!(vi, spl, model)
    end
    vi = DynamicPPL.setindex!!(vi, p, spl)
    vi = DynamicPPL.link!!(vi, spl, model)
    p .= vi[spl]

    # If linking mutated, we need to link once more.
    !linked && DynamicPPL.invlink!!(vi, spl, model)

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
    spl = DynamicPPL.SampleFromPrior()

    ## Check link status of vi
    linked = DynamicPPL.islinked(vi, spl) 
    
    ## transform into unconstrained
    if !linked
        vi = DynamicPPL.link!!(vi, spl, model)
    end
    
    lb = transform(fill(-Inf,length(vi[DynamicPPL.SampleFromPrior()])), vi, model, constrained_space{true}())
    ub = transform(fill(Inf,length(vi[DynamicPPL.SampleFromPrior()])), vi, model, constrained_space{true}())

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
