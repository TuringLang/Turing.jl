module ModeEstimation

using ..Turing
using Bijectors
using Random
using SciMLBase: OptimizationFunction, OptimizationProblem, AbstractADType, NoAD

using DynamicPPL
using DynamicPPL: Model, AbstractContext, VarInfo, VarName,
    _getindex, getsym, getfield, settrans!,  setorder!,
    get_and_set_val!, istrans

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
function DynamicPPL.tilde_assume(rng::Random.AbstractRNG, ctx::OptimizationContext, spl, dist, vn, vi)
    return DynamicPPL.tilde_assume(ctx, spl, dist, vn, vi)
end

function DynamicPPL.tilde_assume(ctx::OptimizationContext{<:LikelihoodContext}, spl, dist, vn, vi)
    r = vi[vn]
    return r, 0, vi
end

function DynamicPPL.tilde_assume(ctx::OptimizationContext, spl, dist, vn, vi)
    r = vi[vn]
    return r, Distributions.logpdf(dist, r), vi
end

# dot assume
function DynamicPPL.dot_tilde_assume(rng::Random.AbstractRNG, ctx::OptimizationContext, sampler, right, left, vns, vi)
    return DynamicPPL.dot_tilde_assume(ctx, sampler, right, left, vns, vi)
end

function DynamicPPL.dot_tilde_assume(ctx::OptimizationContext{<:LikelihoodContext}, sampler::SampleFromPrior, right, left, vns, vi)
    # Values should be set and we're using `SampleFromPrior`, hence the `rng` argument shouldn't
    # affect anything.
    r = DynamicPPL.get_and_set_val!(Random.GLOBAL_RNG, vi, vns, right, sampler)
    return r, 0, vi
end

function DynamicPPL.dot_tilde_assume(ctx::OptimizationContext, sampler::SampleFromPrior, right, left, vns, vi)
    # Values should be set and we're using `SampleFromPrior`, hence the `rng` argument shouldn't
    # affect anything.
    r = DynamicPPL.get_and_set_val!(Random.GLOBAL_RNG, vi, vns, right, sampler)
    return r, loglikelihood(right, r), vi
end

"""
    OptimLogDensity{M<:Model,C<:Context,V<:VarInfo}

A struct that stores the negative log density function of a `DynamicPPL` model.
"""
const OptimLogDensity{M<:Model,C<:OptimizationContext,V<:VarInfo} = Turing.LogDensityFunction{V,M,DynamicPPL.SampleFromPrior,C}

"""
    OptimLogDensity(model::Model, context::OptimizationContext)

Create a callable `OptimLogDensity` struct that evaluates a model using the given `context`.
"""
function OptimLogDensity(model::Model, context::OptimizationContext)
    init = VarInfo(model)
    return Turing.LogDensityFunction(init, model, DynamicPPL.SampleFromPrior(), context)
end

"""
    (f::OptimLogDensity)(z)

Evaluate the negative log joint (with `DefaultContext`) or log likelihood (with `LikelihoodContext`)
at the array `z`.
"""
function (f::OptimLogDensity)(z::AbstractVector)
    sampler = f.sampler
    varinfo = DynamicPPL.VarInfo(f.varinfo, sampler, z)
    return -getlogp(last(DynamicPPL.evaluate!!(f.model, varinfo, sampler, f.context)))
end

function (f::OptimLogDensity)(F, G, z)
    if G !== nothing
        # Calculate negative log joint and its gradient.
        sampler = f.sampler
        neglogp, ∇neglogp = Turing.gradient_logp(
            z, 
            DynamicPPL.VarInfo(f.varinfo, sampler, z),
            f.model, 
            sampler,
            f.context,
        )

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

function transform!(f::OptimLogDensity)
    spl = f.sampler

    ## Check link status of vi in OptimLogDensity
    linked = DynamicPPL.islinked(f.varinfo, spl)

    ## transform into constrained or unconstrained space depending on current state of vi
    if !linked
        DynamicPPL.link!(f.varinfo, spl)
    else
        DynamicPPL.invlink!(f.varinfo, spl)
    end

    return nothing
end

function transform!(p::AbstractArray, vi::DynamicPPL.VarInfo, ::constrained_space{true})
    spl = DynamicPPL.SampleFromPrior()

    linked = DynamicPPL.islinked(vi, spl)
    
    # !linked && DynamicPPL.link!(vi, spl)
    !linked && return identity(p) 
    vi[spl] = p
    DynamicPPL.invlink!(vi,spl)
    p .= vi[spl]

    linked && DynamicPPL.link!(vi,spl)

    return nothing
end

function transform!(p::AbstractArray, vi::DynamicPPL.VarInfo, ::constrained_space{false})
    spl = DynamicPPL.SampleFromPrior()

    linked = DynamicPPL.islinked(vi, spl)
    linked && DynamicPPL.invlink!(vi, spl)
    vi[spl] = p
    DynamicPPL.link!(vi, spl)
    p .= vi[spl]
    !linked && DynamicPPL.invlink!(vi, spl)

    return nothing
end

function transform(p::AbstractArray, vi::DynamicPPL.VarInfo, con::constrained_space)
    tp = copy(p)
    transform!(tp, vi, con)
    return tp
end

abstract type AbstractTransform end

struct ParameterTransform{T<:DynamicPPL.VarInfo, S<:constrained_space} <: AbstractTransform
    vi::T
    space::S
end

struct Init{T<:DynamicPPL.VarInfo, S<:constrained_space} <: AbstractTransform
    vi::T
    space::S
end

function (t::AbstractTransform)(p::AbstractArray)
    return transform(p, t.vi, t.space)
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
    !linked && DynamicPPL.link!(vi, spl)
    
    lb = transform(fill(-Inf,length(vi[DynamicPPL.SampleFromPrior()])), vi, constrained_space{true}())
    ub = transform(fill(Inf,length(vi[DynamicPPL.SampleFromPrior()])), vi, constrained_space{true}())

    return lb, ub
end

function _optim_objective(model::DynamicPPL.Model, ::MAP, ::constrained_space{false})
    ctx = OptimizationContext(DynamicPPL.DefaultContext())
    obj = OptimLogDensity(model, ctx)

    transform!(obj)
    init = Init(obj.varinfo, constrained_space{false}())
    t = ParameterTransform(obj.varinfo, constrained_space{true}())

    return (obj=obj, init = init, transform=t)
end

function _optim_objective(model::DynamicPPL.Model, ::MAP, ::constrained_space{true})
    ctx = OptimizationContext(DynamicPPL.DefaultContext())
    obj = OptimLogDensity(model, ctx)
    
    init = Init(obj.varinfo, constrained_space{true}())
    t = ParameterTransform(obj.varinfo, constrained_space{true}())
    
    return (obj=obj, init = init, transform=t)
end

function _optim_objective(model::DynamicPPL.Model, ::MLE,  ::constrained_space{false})
    ctx = OptimizationContext(DynamicPPL.LikelihoodContext())
    obj = OptimLogDensity(model, ctx)
    
    transform!(obj)
    init = Init(obj.varinfo, constrained_space{false}())
    t = ParameterTransform(obj.varinfo, constrained_space{true}())
    
    return (obj=obj, init = init, transform=t)
end

function _optim_objective(model::DynamicPPL.Model, ::MLE, ::constrained_space{true})
    ctx = OptimizationContext(DynamicPPL.LikelihoodContext())
    obj = OptimLogDensity(model, ctx)
  
    init = Init(obj.varinfo, constrained_space{true}())
    t = ParameterTransform(obj.varinfo, constrained_space{true}())
    
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
