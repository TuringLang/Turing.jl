module ModeEstimation

using ..Turing
using Bijectors
using DiffEqBase: OptimizationFunction

using DynamicPPL
import DynamicPPL: Model, AbstractContext, VarInfo, VarName,
    _getindex, getsym, getfield, settrans!,  setorder!,
    get_and_set_val!, istrans, tilde, dot_tilde, get_vns_and_dist

export  MAP,
        MLE,
        OptimLogDensity,
        OptimizationContext,
        instantiate_optimisation_problem, 
        instantiate_galacticoptim_function


struct MLE{constrained} end
struct MAP{constrained} end

MLE(constrained) = MLE{constrained}()
MAP(constrained) = MAP{constrained}()
MLE() = MLE{false}()
MAP() = MAP{false}()

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
function DynamicPPL.tilde(rng, ctx::OptimizationContext, spl, dist, vn::VarName, inds, vi)
    return DynamicPPL.tilde(ctx, spl, dist, vn, inds, vi)
end

function DynamicPPL.tilde(ctx::OptimizationContext{<:LikelihoodContext}, spl, dist, vn::VarName, inds, vi)
    r = vi[vn]
    return r, 0
end

function DynamicPPL.tilde(ctx::OptimizationContext, spl, dist, vn::VarName, inds, vi)
    r = vi[vn]
    return r, Distributions.logpdf(dist, r)
end


# observe
function DynamicPPL.tilde(rng, ctx::OptimizationContext, sampler, right, left, vi)
    return DynamicPPL.tilde(ctx, sampler, right, left, vi)
end

function DynamicPPL.tilde(ctx::OptimizationContext{<:PriorContext}, sampler, right, left, vi)
    return 0
end

function DynamicPPL.tilde(ctx::OptimizationContext, sampler, dist, value, vi)
    return Distributions.logpdf(dist, value)
end

# dot assume
function DynamicPPL.dot_tilde(rng, ctx::OptimizationContext, sampler, right, left, vn::VarName, inds, vi)
    return DynamicPPL.dot_tilde(ctx, sampler, right, left, vn, inds, vi)
end

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


function transform2constrained!(p::AbstractArray, vi::DynamicPPL.VarInfo)
  spl = DynamicPPL.SampleFromPrior()

  linked = DynamicPPL.islinked(vi, spl)
  
  !linked && DynamicPPL.link!(vi, spl)
  vi[spl] = p
  DynamicPPL.invlink!(vi,spl)
  p .= vi[spl]
  
  linked && DynamicPPL.link!(vi,spl)

  return nothing
end

function transform2constrained(p::AbstractArray, vi::DynamicPPL.VarInfo)
    tp = copy(p)
    transform2constrained!(tp, vi)
  return tp
end

function transform2unconstrained!(p::AbstractArray, vi::DynamicPPL.VarInfo)
  spl = DynamicPPL.SampleFromPrior()

  linked = DynamicPPL.islinked(vi, spl)
  linked && DynamicPPL.invlink!(vi, spl)
  vi[spl] = p
  DynamicPPL.link!(vi, spl)
  p .= vi[spl]
  !linked && DynamicPPL.invlink!(vi, spl)

  return nothing
end

function transform2unconstrained(p::AbstractArray, vi::DynamicPPL.VarInfo)
  tp = copy(p)
  transform2unconstrained!(tp, vi)
  return tp
end

abstract type AbstractTransformFunction end
abstract type AbstractParameterTransformFunction <: AbstractTransformFunction end
abstract type AbstractInitTransformFunction <: AbstractParameterTransformFunction end


struct ParameterTransformFunction <: AbstractParameterTransformFunction
    vi::DynamicPPL.VarInfo
    transform
end

struct InitTransformFunction <: AbstractInitTransformFunction
    vi::DynamicPPL.VarInfo
    transform
end

function (t::ParameterTransformFunction)(p::AbstractArray)
    return t.transform(p, t.vi)
end 

function (t::InitTransformFunction)(p::AbstractArray)
    return t.transform(p, t.vi)
end 

function (t::InitTransformFunction)()
    return t.vi[DynamicPPL.SampleFromPrior()]
end 


function instantiate_optimisation_problem(model::DynamicPPL.Model, ::MAP{false})
  ctx = OptimizationContext(DynamicPPL.DefaultContext())
  obj = OptimLogDensity(model, ctx)

  transform!(obj)
  init = InitTransformFunction(obj.vi, transform2unconstrained)
  t = ParameterTransformFunction(obj.vi, transform2constrained)

  return (obj=obj, init = init, transform=t)
end

function instantiate_optimisation_problem(model::DynamicPPL.Model, ::MAP{true})
    ctx = OptimizationContext(DynamicPPL.DefaultContext())
    obj = OptimLogDensity(model, ctx)
  
    init = InitTransformFunction(obj.vi, (init_vals::AbstractArray, vi) -> identity(init_vals))
    t = ParameterTransformFunction(obj.vi, (p::AbstractArray, vi) -> identity(p))
      
    return (obj=obj, init = init, transform=t)
  end

function instantiate_optimisation_problem(model::DynamicPPL.Model, ::MLE{false})
    ctx = OptimizationContext(DynamicPPL.LikelihoodContext())
    obj = OptimLogDensity(model, ctx)
  
    transform!(obj)
    init = InitTransformFunction(obj.vi, transform2unconstrained)
    t = ParameterTransformFunction(obj.vi, transform2constrained)
  
    return (obj=obj, init = init, transform=t)
end

function instantiate_optimisation_problem(model::DynamicPPL.Model, ::MLE{true})
    ctx = OptimizationContext(DynamicPPL.LikelihoodContext())
    obj = OptimLogDensity(model, ctx)
  
    init = InitTransformFunction(obj.vi, (init_vals::AbstractArray, vi) -> identity(init_vals))
    t = ParameterTransformFunction(obj.vi, (p::AbstractArray, vi) -> identity(p))
      
    return (obj=obj, init = init, transform=t)
end

function instantiate_galacticoptim_function(model::DynamicPPL.Model, estimator::Union{MLE,MAP})
  obj, init, t = instantiate_optimisation_problem(model, estimator)
  
  l(x,p) = obj(x)
  f = OptimizationFunction(l; grad = (G,x,p) -> obj(nothing, G, nothing, x), hess = (H,x,p) -> obj(nothing, nothing, H, x))

  return (f=f, init=init, transform = t)
end

end