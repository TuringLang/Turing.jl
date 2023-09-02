module TuringOptimExt

if isdefined(Base, :get_extension)
    import Turing
    import Turing: Distributions, DynamicPPL, ForwardDiff, NamedArrays, Printf, Setfield, Statistics, StatsAPI, StatsBase 
    import Optim
else
    import ..Turing
    import ..Turing: Distributions, DynamicPPL, ForwardDiff, NamedArrays, Printf, Setfield, Statistics, StatsAPI, StatsBase
    import ..Optim
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
    M<:Turing.OptimLogDensity
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
    p = map(z -> StatsAPI.pvalue(Distributions.Normal(), z; tail=:both), zscore)

    # Confidence interval (CI)
    q = Statistics.quantile(Distributions.Normal(), (1 + level) / 2)
    ci_low = estimates .- q .* stderrors
    ci_high = estimates .+ q .* stderrors
    
    level_ = 100*level
    level_percentage = isinteger(level_) ? Int(level_) : level_ 
    
    StatsBase.CoefTable(
        [estimates, stderrors, zscore, p, ci_low, ci_high],
        ["Coef.", "Std. Error", "z", "Pr(>|z|)", "Lower $(level_percentage)%", "Upper $(level_percentage)%"],
        terms)
end

function StatsBase.informationmatrix(m::ModeResult; hessian_function=ForwardDiff.hessian, kwargs...)
    # Calculate Hessian and information matrix.

    # Convert the values to their unconstrained states to make sure the
    # Hessian is computed with respect to the untransformed parameters.
    linked = DynamicPPL.istrans(m.f.varinfo)
    if linked
        Setfield.@set! m.f.varinfo = DynamicPPL.invlink!!(m.f.varinfo, m.f.model)
    end

    # Calculate the Hessian.
    varnames = StatsBase.coefnames(m)
    H = hessian_function(m.f, m.values.array[:, 1])
    info = inv(H)

    # Link it back if we invlinked it.
    if linked
        Setfield.@set! m.f.varinfo = DynamicPPL.link!!(m.f.varinfo, m.f.model)
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
function Optim.optimize(model::DynamicPPL.Model, ::Turing.MLE, options::Optim.Options=Optim.Options(); kwargs...)
    ctx = Turing.OptimizationContext(DynamicPPL.LikelihoodContext())
    f = Turing.OptimLogDensity(model, ctx)
    init_vals = DynamicPPL.getparams(f)
    optimizer = Optim.LBFGS()
    return _mle_optimize(model, init_vals, optimizer, options; kwargs...)
end
function Optim.optimize(model::DynamicPPL.Model, ::Turing.MLE, init_vals::AbstractArray, options::Optim.Options=Optim.Options(); kwargs...)
    optimizer = Optim.LBFGS()
    return _mle_optimize(model, init_vals, optimizer, options; kwargs...)
end
function Optim.optimize(model::DynamicPPL.Model, ::Turing.MLE, optimizer::Optim.AbstractOptimizer, options::Optim.Options=Optim.Options(); kwargs...)
    ctx = Turing.OptimizationContext(DynamicPPL.LikelihoodContext())
    f = Turing.OptimLogDensity(model, ctx)
    init_vals = DynamicPPL.getparams(f)
    return _mle_optimize(model, init_vals, optimizer, options; kwargs...)
end
function Optim.optimize(
    model::DynamicPPL.Model,
    ::Turing.MLE,
    init_vals::AbstractArray,
    optimizer::Optim.AbstractOptimizer,
    options::Optim.Options=Optim.Options();
    kwargs...
)
    return _mle_optimize(model, init_vals, optimizer, options; kwargs...)
end

function _mle_optimize(model::DynamicPPL.Model, args...; kwargs...)
    ctx = Turing.OptimizationContext(DynamicPPL.LikelihoodContext())
    return _optimize(model, Turing.OptimLogDensity(model, ctx), args...; kwargs...)
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

function Optim.optimize(model::DynamicPPL.Model, ::Turing.MAP, options::Optim.Options=Optim.Options(); kwargs...)
    ctx = Turing.OptimizationContext(DynamicPPL.DefaultContext())
    f = Turing.OptimLogDensity(model, ctx)
    init_vals = DynamicPPL.getparams(f)
    optimizer = Optim.LBFGS()
    return _map_optimize(model, init_vals, optimizer, options; kwargs...)
end
function Optim.optimize(model::DynamicPPL.Model, ::Turing.MAP, init_vals::AbstractArray, options::Optim.Options=Optim.Options(); kwargs...)
    optimizer = Optim.LBFGS()
    return _map_optimize(model, init_vals, optimizer, options; kwargs...)
end
function Optim.optimize(model::DynamicPPL.Model, ::Turing.MAP, optimizer::Optim.AbstractOptimizer, options::Optim.Options=Optim.Options(); kwargs...)
    ctx = Turing.OptimizationContext(DynamicPPL.DefaultContext())
    f = Turing.OptimLogDensity(model, ctx)
    init_vals = DynamicPPL.getparams(f)
    return _map_optimize(model, init_vals, optimizer, options; kwargs...)
end
function Optim.optimize(
    model::DynamicPPL.Model,
    ::Turing.MAP,
    init_vals::AbstractArray,
    optimizer::Optim.AbstractOptimizer,
    options::Optim.Options=Optim.Options();
    kwargs...
)
    return _map_optimize(model, init_vals, optimizer, options; kwargs...)
end

function _map_optimize(model::DynamicPPL.Model, args...; kwargs...)
    ctx = Turing.OptimizationContext(DynamicPPL.DefaultContext())
    return _optimize(model, Turing.OptimLogDensity(model, ctx), args...; kwargs...)
end

"""
    _optimize(model::Model, f::OptimLogDensity, optimizer=Optim.LBFGS(), args...; kwargs...)

Estimate a mode, i.e., compute a MLE or MAP estimate.
"""
function _optimize(
    model::DynamicPPL.Model,
    f::Turing.OptimLogDensity,
    init_vals::AbstractArray=DynamicPPL.getparams(f),
    optimizer::Optim.AbstractOptimizer=Optim.LBFGS(),
    options::Optim.Options=Optim.Options(),
    args...;
    kwargs...
)
    # Convert the initial values, since it is assumed that users provide them
    # in the constrained space.
    Setfield.@set! f.varinfo = DynamicPPL.unflatten(f.varinfo, init_vals)
    Setfield.@set! f.varinfo = DynamicPPL.link!!(f.varinfo, model)
    init_vals = DynamicPPL.getparams(f)

    # Optimize!
    M = Optim.optimize(Optim.only_fg!(f), init_vals, optimizer, options, args...; kwargs...)

    # Warn the user if the optimization did not converge.
    if !Optim.converged(M)
        @warn "Optimization did not converge! You may need to correct your model or adjust the Optim parameters."
    end

    # Get the VarInfo at the MLE/MAP point, and run the model to ensure
    # correct dimensionality.
    Setfield.@set! f.varinfo = DynamicPPL.unflatten(f.varinfo, M.minimizer)
    Setfield.@set! f.varinfo = DynamicPPL.invlink!!(f.varinfo, model)
    vals = DynamicPPL.getparams(f)
    Setfield.@set! f.varinfo = DynamicPPL.link!!(f.varinfo, model)

    # Make one transition to get the parameter names.
    ts = [Turing.Inference.Transition(
        Turing.Inference.getparams(model, f.varinfo),
        DynamicPPL.getlogp(f.varinfo)
    )]
    varnames, _ = Turing.Inference._params_to_array(model, ts)

    # Store the parameters and their names in an array.
    vmat = NamedArrays.NamedArray(vals, varnames)

    return ModeResult(vmat, M, -M.minimum, f)
end

end # module
