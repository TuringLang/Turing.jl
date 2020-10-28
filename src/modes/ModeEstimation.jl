using ..Turing
using ..Bijectors
using LinearAlgebra

import ..AbstractMCMC: AbstractSampler
import ..DynamicPPL
import ..DynamicPPL: Model, AbstractContext, VarInfo, AbstractContext, VarName,
    _getindex, getsym, getfield, settrans!,  setorder!,
    get_and_set_val!, istrans, tilde, dot_tilde, get_vns_and_dist
import .Optim
import .Optim: optimize
import ..ForwardDiff
import NamedArrays
import StatsBase
import Printf

struct MLE end
struct MAP end

struct constrained end
struct unconstrained end
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
        l, g = gradient_logp(
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
    print(io, "ModeResult with maximized lp of ")
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

    # Convert the values to their unconstrained states to make sure the
    # Hessian is computed with respect to the untransformed parameters.
    spl = DynamicPPL.SampleFromPrior()

    # NOTE: This should be converted to islinked(vi, spl) after
    # https://github.com/TuringLang/DynamicPPL.jl/pull/124 goes through.
    vns = DynamicPPL._getvns(m.f.vi, spl)
    
    linked = DynamicPPL._islinked(m.f.vi, vns)
    linked && invlink!(m.f.vi, spl)

    # Calculate the Hessian.
    varnames = StatsBase.coefnames(m)
    H = hessian_function(m.f, m.values.array[:, 1])
    info = inv(H)

    # Link it back if we invlinked it.
    linked && link!(m.f.vi, spl)

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
function Optim.optimize(model::Model, ::MLE, options::Optim.Options=Optim.Options(); kwargs...) 
    return _mle_optimize(model, options; kwargs...)
end
function Optim.optimize(model::Model, ::MLE, init_vals::AbstractArray, options::Optim.Options=Optim.Options(); kwargs...)
    return _mle_optimize(model, init_vals, options; kwargs...)
end
function Optim.optimize(model::Model, ::MLE, optimizer::Optim.AbstractOptimizer, options::Optim.Options=Optim.Options(); kwargs...)
    return _mle_optimize(model, optimizer, options; kwargs...)
end
function Optim.optimize(
    model::Model, 
    ::MLE, 
    init_vals::AbstractArray, 
    optimizer::Optim.AbstractOptimizer, 
    options::Optim.Options=Optim.Options(); 
    kwargs...
)
    return _mle_optimize(model, init_vals, optimizer, options; kwargs...)
end

function _mle_optimize(model::Model, args...; kwargs...)
    ctx = OptimizationContext(DynamicPPL.LikelihoodContext())
    return _optimize(model, OptimLogDensity(model, ctx), args...; kwargs...)
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

function Optim.optimize(model::Model, ::MAP, options::Optim.Options=Optim.Options(); kwargs...) 
    return _map_optimize(model, options; kwargs...)
end
function Optim.optimize(model::Model, ::MAP, init_vals::AbstractArray, options::Optim.Options=Optim.Options(); kwargs...)
    return _map_optimize(model, init_vals, options; kwargs...)
end
function Optim.optimize(model::Model, ::MAP, optimizer::Optim.AbstractOptimizer, options::Optim.Options=Optim.Options(); kwargs...)
    return _map_optimize(model, optimizer, options; kwargs...)
end
function Optim.optimize(
    model::Model, 
    ::MAP, 
    init_vals::AbstractArray, 
    optimizer::Optim.AbstractOptimizer, 
    options::Optim.Options=Optim.Options(); 
    kwargs...
)
    return _map_optimize(model, init_vals, optimizer, options; kwargs...)
end

function _map_optimize(model::Model, args...; kwargs...)
    ctx = OptimizationContext(DynamicPPL.DefaultContext())
    return _optimize(model, OptimLogDensity(model, ctx), args...; kwargs...)
end

"""
    _optimize(model::Model, f::OptimLogDensity, optimizer=Optim.LBFGS(), args...; kwargs...)

Estimate a mode, i.e., compute a MLE or MAP estimate.
"""
function _optimize(
    model::Model, 
    f::OptimLogDensity, 
    optimizer::Optim.AbstractOptimizer = Optim.LBFGS(),
    args...; 
    kwargs...
)
    return _optimize(model, f, f.vi[DynamicPPL.SampleFromPrior()], optimizer, args...; kwargs...)
end

function _optimize(
    model::Model, 
    f::OptimLogDensity, 
    options::Optim.Options = Optim.Options(),
    args...; 
    kwargs...
)
    return _optimize(model, f, f.vi[DynamicPPL.SampleFromPrior()], Optim.LBFGS(), args...; kwargs...)
end

function _optimize(
    model::Model, 
    f::OptimLogDensity, 
    init_vals::AbstractArray = f.vi[DynamicPPL.SampleFromPrior()], 
    options::Optim.Options = Optim.Options(),
    args...; 
    kwargs...
)
    return _optimize(model, f,init_vals, Optim.LBFGS(), options, args...; kwargs...)
end

function _optimize(
    model::Model, 
    f::OptimLogDensity, 
    init_vals::AbstractArray = f.vi[DynamicPPL.SampleFromPrior()], 
    optimizer::Optim.AbstractOptimizer = Optim.LBFGS(),
    options::Optim.Options = Optim.Options(),
    args...; 
    kwargs...
)
    # Do some initialization.
    spl = DynamicPPL.SampleFromPrior()

    # Convert the initial values, since it is assumed that users provide them
    # in the constrained space.
    f.vi[spl] = init_vals
    link!(f.vi, spl)
    init_vals = f.vi[spl]

    # Optimize!
    M = Optim.optimize(Optim.only_fgh!(f), init_vals, optimizer, options, args...; kwargs...)

    # Warn the user if the optimization did not converge.
    if !Optim.converged(M)
        @warn "Optimization did not converge! You may need to correct your model or adjust the Optim parameters."
    end

    # Get the VarInfo at the MLE/MAP point, and run the model to ensure 
    # correct dimensionality.
    f.vi[spl] = M.minimizer
    invlink!(f.vi, spl)
    vals = f.vi[spl]
    link!(f.vi, spl)

    # Make one transition to get the parameter names.
    ts = [Turing.Inference.Transition(DynamicPPL.tonamedtuple(f.vi), DynamicPPL.getlogp(f.vi))]
    varnames, _ = Turing.Inference._params_to_array(ts)

    # Store the parameters and their names in an array.
    vmat = NamedArrays.NamedArray(vals, varnames)

    return ModeResult(vmat, M, -M.minimum, f)
end


#################################################
# Generic optimisation objective initialisation #
#################################################

function (f::Turing.OptimLogDensity)(G, z)
  spl = DynamicPPL.SampleFromPrior()
  
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

  return nothing
end



function transform!(f::OptimLogDensity)
  spl = DynamicPPL.SampleFromPrior()

  ## Check link status of vi in OptimLogDensity
  linked = DynamicPPL.islinked(f.vi, spl) 

  ## transform into constrained or unconstrained space depending on current state of vi
  if !linked
    f.vi[spl] = f.vi[DynamicPPL.SampleFromPrior()]
    DynamicPPL.link!(f.vi, spl)
  else
    DynamicPPL.invlink!(f.vi, spl)
  end

  return nothing
end


function transform2constrained(par::AbstractArray, vi::DynamicPPL.VarInfo)
  spl = DynamicPPL.SampleFromPrior()

  linked = DynamicPPL.islinked(vi, spl)
  
  !linked && DynamicPPL.link!(vi, spl)
  vi[spl] = par
  DynamicPPL.invlink!(vi,spl)
  tpar = vi[spl]
  
  linked && DynamicPPL.link!(vi,spl)

  return tpar
end

function transform2constrained!(par::AbstractArray, vi::DynamicPPL.VarInfo)
  par .= transform2constrained(par, vi)
  return nothing
end

function transform2unconstrained(par::AbstractArray, vi::DynamicPPL.VarInfo)
  spl = DynamicPPL.SampleFromPrior()

  linked = DynamicPPL.islinked(vi, spl)
  linked && DynamicPPL.invlink!(vi, spl)
  vi[spl] = par
  DynamicPPL.link!(vi, spl)
  tpar = vi[spl]
  !linked && DynamicPPL.invlink!(vi, spl)

  return tpar
end

function transform2unconstrained!(par::AbstractArray, vi::DynamicPPL.VarInfo)
  par .= transform2unconstrained(par, vi)
  return nothing
end


function _orderpar(par::NamedTuple, vi::DynamicPPL.VarInfo)
  tmp_idx = indexin(collect(keys(vi.metadata)), collect(keys(par)))
  tmp = collect(par)[tmp_idx]
  return tmp, tmp_idx
end


function transform2unconstrained(par::NamedTuple, vi::DynamicPPL.VarInfo; order::Bool=true, array::Bool=true)
  par_sor, par_idx = _orderpar(par, vi)

  transform2unconstrained!(par_sor, vi)

  if order
    if array
      return par_sor
    else
      return (; zip(keys(vi.metadata), par_sor)...)
    end
  else
    if array
      return par_sor[tmp_idx]
    else
      return (; zip(keys(par),par_sor[par_idx])...)
    end
  end
end

function transform2constrained(par::NamedTuple, vi::DynamicPPL.VarInfo; order::Bool=true, array::Bool=true)
  par_sor, par_idx = _orderpar(par, vi)

  transform2constrained!(par_sor, vi)

  if order
    if array
      return par_sor
    else
      return (; zip(keys(vi.metadata), par_sor)...)
    end
  else
    if array
      return par_sor[tmp_idx]
    else
      return (; zip(keys(par),par_sor[par_idx])...)
    end
  end
end


function transform2constrained(res::Optim.MultivariateOptimizationResults, vi::DynamicPPL.VarInfo)
  tres = deepcopy(res)
  if !any(isnan.(tres.minimizer))
    tres.minimizer = transform2constrained(tres.minimizer, vi)
  else
      @warn "Could not transform optimisation results due to NaNs in ':minimizer'."
  end

  if !any(isnan.(tres.initial_x))
    tres.initial_x = transform2constrained(tres.initial_x, vi)
  else
    @warn "Could not transform initial values due to NaNs in ':initial_x'."
  end

  return tres
end



function instantiate_optimisation_problem(model::DynamicPPL.Model, ::MAP , ::unconstrained)
  obj = OptimLogDensity(model, OptimizationContext(DynamicPPL.DefaultContext()))

  transform!(obj)

  init(init_vals) = transform2unconstrained(init_vals, obj.vi)
  t(res) = transform2constrained(res, obj.vi)

  return (obj=obj, init = init, transform=t)
end

function instantiate_optimisation_problem(model::DynamicPPL.Model, ::MAP , ::constrained)
  obj = OptimLogDensity(model, OptimizationContext(DynamicPPL.DefaultContext()))

  init(init_vals) = isa(init_vals, NamedTuple) ? _orderpar(init_vals, obj.vi)[1] : identity(init_vals) 

  return (obj=obj, init = init, transform=identity)
end

function instantiate_optimisation_problem(model::DynamicPPL.Model, ::MLE , ::unconstrained)
  obj = OptimLogDensity(model, OptimizationContext(DynamicPPL.LikelihoodContext()))

  transform!(obj)

  init(init_vals) = transform2unconstrained(init_vals, obj.vi)
  t(res) = transform2constrained(res, obj.vi)

  return (obj=obj, init = init, transform=t)
end

function instantiate_optimisation_problem(model::DynamicPPL.Model, ::MLE, ::constrained)
  obj = OptimLogDensity(model, OptimizationContext(DynamicPPL.LikelihoodContext()))

  init(init_vals) = isa(init_vals, NamedTuple) ? _orderpar(init_vals, obj.vi)[1] : identity(init_vals) 

  return (obj=obj, init = init, transform=identity)
end