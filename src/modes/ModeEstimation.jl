module ModeEstimation

using ..Turing
using Bijectors
using Random
using SciMLBase: OptimizationFunction, OptimizationProblem, AbstractADType

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
    return r, 0
end

function DynamicPPL.tilde_assume(ctx::OptimizationContext, spl, dist, vn, vi)
    r = vi[vn]
    return r, Distributions.logpdf(dist, r)
end

# dot assume
function DynamicPPL.dot_tilde_assume(rng::Random.AbstractRNG, ctx::OptimizationContext, sampler, right, left, vns, vi)
    return DynamicPPL.dot_tilde_assume(ctx, sampler, right, left, vns, vi)
end

function DynamicPPL.dot_tilde_assume(ctx::OptimizationContext{<:LikelihoodContext}, sampler::SampleFromPrior, right, left, vns, vi)
    # Values should be set and we're using `SampleFromPrior`, hence the `rng` argument shouldn't
    # affect anything.
    r = DynamicPPL.get_and_set_val!(Random.GLOBAL_RNG, vi, vns, right, sampler)
    return r, 0
end

function DynamicPPL.dot_tilde_assume(ctx::OptimizationContext, sampler::SampleFromPrior, right, left, vns, vi)
    # Values should be set and we're using `SampleFromPrior`, hence the `rng` argument shouldn't
    # affect anything.
    r = DynamicPPL.get_and_set_val!(Random.GLOBAL_RNG, vi, vns, right, sampler)
    return r, loglikelihood(right, r)
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

function (f::OptimLogDensity)(F, G, H, z)
    # Throw an error if a second order method was used.
    if H !== nothing
        error("Second order optimization is not yet supported.")
    end

    spl = DynamicPPL.SampleFromPrior()
    
    if G !== nothing
        # Calculate log joint and the gradient
        l, g = Turing.gradient_logp(
            z, 
            DynamicPPL.VarInfo(f.vi, spl, z), 
            f.model, 
            spl,
            f.context
        )

        # Use the negative gradient because we are minimizing.
        G[:] = -g

        # If F is something, return that since we already have the 
        # log joint.
        if F !== nothing
            F = -l
            return F
        end
    end

    # No gradient necessary, just return the log joint.
    if F !== nothing
        F = f(z)
        return F
    end

    return nothing
end



#################################################
# Generic optimisation objective initialisation #
#################################################

function transform!(f::OptimLogDensity)
  spl = DynamicPPL.SampleFromPrior()

  ## Check link status of vi in OptimLogDensity
  linked = DynamicPPL.islinked(f.vi, spl) 

  ## transform into constrained or unconstrained space depending on current state of vi
  if !linked
    DynamicPPL.link!(f.vi, spl)
  else
    DynamicPPL.invlink!(f.vi, spl)
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
  init = Init(obj.vi, constrained_space{false}())
  t = ParameterTransform(obj.vi, constrained_space{true}())

  return (obj=obj, init = init, transform=t)
end

function _optim_objective(model::DynamicPPL.Model, ::MAP, ::constrained_space{true})
    ctx = OptimizationContext(DynamicPPL.DefaultContext())
    obj = OptimLogDensity(model, ctx)
  
    init = Init(obj.vi, constrained_space{true}())
    t = ParameterTransform(obj.vi, constrained_space{true}())
      
    return (obj=obj, init = init, transform=t)
  end

function _optim_objective(model::DynamicPPL.Model, ::MLE,  ::constrained_space{false})
    ctx = OptimizationContext(DynamicPPL.LikelihoodContext())
    obj = OptimLogDensity(model, ctx)
  
    transform!(obj)
    init = Init(obj.vi, constrained_space{false}())
    t = ParameterTransform(obj.vi, constrained_space{true}())
  
    return (obj=obj, init = init, transform=t)
end

function _optim_objective(model::DynamicPPL.Model, ::MLE, ::constrained_space{true})
    ctx = OptimizationContext(DynamicPPL.LikelihoodContext())
    obj = OptimLogDensity(model, ctx)
  
    init = Init(obj.vi, constrained_space{true}())
    t = ParameterTransform(obj.vi, constrained_space{true}())
    
    return (obj=obj, init = init, transform=t)
end

function optim_objective(model::DynamicPPL.Model, estimator::Union{MLE, MAP}; constrained::Bool=true)
    return _optim_objective(model, estimator, constrained_space{constrained}())
end


function optim_function(model::DynamicPPL.Model, estimator::Union{MLE, MAP}; constrained::Bool=true, autoad::Union{Nothing, AbstractADType}=nothing)
    obj, init, t = optim_objective(model, estimator; constrained=constrained)
  
    l(x,p) = obj(x)
    f = isa(autoad, AbstractADType) ? OptimizationFunction(l, autoad) : OptimizationFunction(l; grad = (G,x,p) -> obj(nothing, G, nothing, x), hess = (H,x,p) -> obj(nothing, nothing, H, x))
  
    return (func=f, init=init, transform = t)
end


function optim_problem(model::DynamicPPL.Model, estimator::Union{MAP, MLE}; constrained::Bool=true, init_theta=nothing, autoad::Union{Nothing, AbstractADType}=nothing, kwargs...)
    f = optim_function(model, estimator; constrained=constrained, autoad=autoad)

    init_theta = init_theta === nothing ? f.init() : f.init(init_theta)

    prob = OptimizationProblem(f.func, init_theta, nothing; kwargs...)

    return (prob=prob, init=f.init, transform = f.transform)
end

end
