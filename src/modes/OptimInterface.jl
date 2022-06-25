using DynamicPPL: DefaultContext, LikelihoodContext
import .Optim
import .Optim: optimize
import ..ForwardDiff
import NamedArrays
import StatsBase
import Printf


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
    return _optimize(model, f, f.varinfo[f.sampler], optimizer, args...; kwargs...)
end

function _optimize(
    model::Model, 
    f::OptimLogDensity, 
    options::Optim.Options = Optim.Options(),
    args...; 
    kwargs...
)
    return _optimize(model, f, f.varinfo[f.sampler], Optim.LBFGS(), args...; kwargs...)
end

function _optimize(
    model::Model, 
    f::OptimLogDensity, 
    init_vals::AbstractArray = f.varinfo[f.sampler],
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
    spl = f.sampler

    # Convert the initial values, since it is assumed that users provide them
    # in the constrained space.
    f.varinfo[spl] = init_vals
    link!(f.varinfo, spl)
    init_vals = f.varinfo[spl]

    # Optimize!
    M = Optim.optimize(Optim.only_fgh!(f), init_vals, optimizer, options, args...; kwargs...)

    # Warn the user if the optimization did not converge.
    if !Optim.converged(M)
        @warn "Optimization did not converge! You may need to correct your model or adjust the Optim parameters."
    end

    # Get the VarInfo at the MLE/MAP point, and run the model to ensure 
    # correct dimensionality.
    f.varinfo[spl] = M.minimizer
    invlink!(f.varinfo, spl)
    vals = f.varinfo[spl]
    link!(f.varinfo, spl)

    # Make one transition to get the parameter names.
    ts = [Turing.Inference.Transition(DynamicPPL.tonamedtuple(f.varinfo), DynamicPPL.getlogp(f.varinfo))]
    varnames, _ = Turing.Inference._params_to_array(ts)

    # Store the parameters and their names in an array.
    vmat = NamedArrays.NamedArray(vals, varnames)

    return ModeResult(vmat, M, -M.minimum, f)
end
