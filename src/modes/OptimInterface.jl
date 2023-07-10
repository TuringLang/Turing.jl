using Setfield
using DynamicPPL: DefaultContext, LikelihoodContext
using DynamicPPL: DynamicPPL
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
    values::V
    "The stored Optim.jl results."
    optim_result::O
    "The final log likelihood or log joint, depending on whether `MAP` or `MLE` was run."
    lp::Float64
    "The evaluation function used to calculate the output."
    f::M
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

function StatsBase.coeftable(m::ModeResult; level::Real=0.95)
    # Get columns for coeftable.
    terms = string.(StatsBase.coefnames(m))
    estimates = m.values.array[:, 1]
    stderrors = StatsBase.stderror(m)
    zscore = estimates ./ stderrors
    p = pvalue(Normal(), zscore; tail=:both)

    # Confidence interval (CI)
    q = quantile(Normal(), (1 + level) / 2)
    ci_low = estimates .- q .* stderrors
    ci_high = estimates .+ q .* stderrors

    StatsBase.CoefTable(
        [estimates, stderrors, zscore, p, ci_low, ci_high],
        ["Coef.", "Std. Error", "z", "Pr(>|z|)", "Lower 95%", "Upper 95%"],
        terms)
end

function StatsBase.informationmatrix(m::ModeResult; hessian_function=ForwardDiff.hessian, kwargs...)
    # Calculate Hessian and information matrix.

    # Convert the values to their unconstrained states to make sure the
    # Hessian is computed with respect to the untransformed parameters.
    linked = DynamicPPL.istrans(m.f.varinfo)
    if linked
        @set! m.f.varinfo = invlink!!(m.f.varinfo, m.f.model)
    end

    # Calculate the Hessian.
    varnames = StatsBase.coefnames(m)
    H = hessian_function(m.f, m.values.array[:, 1])
    info = inv(H)

    # Link it back if we invlinked it.
    if linked
        @set! m.f.varinfo = link!!(m.f.varinfo, m.f.model)
    end

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
    optimizer::Optim.AbstractOptimizer=Optim.LBFGS(),
    args...;
    kwargs...
)
    return _optimize(model, f, DynamicPPL.getparams(f), optimizer, args...; kwargs...)
end

function _optimize(
    model::Model,
    f::OptimLogDensity,
    options::Optim.Options=Optim.Options(),
    args...;
    kwargs...
)
    return _optimize(model, f, DynamicPPL.getparams(f), Optim.LBFGS(), args...; kwargs...)
end

function _optimize(
    model::Model,
    f::OptimLogDensity,
    init_vals::AbstractArray=DynamicPPL.getparams(f),
    options::Optim.Options=Optim.Options(),
    args...;
    kwargs...
)
    return _optimize(model, f, init_vals, Optim.LBFGS(), options, args...; kwargs...)
end

function _optimize(
    model::Model,
    f::OptimLogDensity,
    init_vals::AbstractArray=DynamicPPL.getparams(f),
    optimizer::Optim.AbstractOptimizer=Optim.LBFGS(),
    options::Optim.Options=Optim.Options(),
    args...;
    kwargs...
)
    # Convert the initial values, since it is assumed that users provide them
    # in the constrained space.
    @set! f.varinfo = DynamicPPL.unflatten(f.varinfo, init_vals)
    @set! f.varinfo = DynamicPPL.link!!(f.varinfo, model)
    init_vals = DynamicPPL.getparams(f)

    # Optimize!
    M = Optim.optimize(Optim.only_fg!(f), init_vals, optimizer, options, args...; kwargs...)

    # Warn the user if the optimization did not converge.
    if !Optim.converged(M)
        @warn "Optimization did not converge! You may need to correct your model or adjust the Optim parameters."
    end

    # Get the VarInfo at the MLE/MAP point, and run the model to ensure
    # correct dimensionality.
    @set! f.varinfo = DynamicPPL.unflatten(f.varinfo, M.minimizer)
    @set! f.varinfo = invlink!!(f.varinfo, model)
    vals = DynamicPPL.getparams(f)
    @set! f.varinfo = link!!(f.varinfo, model)

    # Make one transition to get the parameter names.
    ts = [Turing.Inference.Transition(
        DynamicPPL.tonamedtuple(f.varinfo),
        DynamicPPL.getlogp(f.varinfo)
    )]
    varnames, _ = Turing.Inference._params_to_array(ts)

    # Store the parameters and their names in an array.
    vmat = NamedArrays.NamedArray(vals, varnames)

    return ModeResult(vmat, M, -M.minimum, f)
end
