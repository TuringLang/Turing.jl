module TuringOptimExt

if isdefined(Base, :get_extension)
    import Turing
    import Turing:
        DynamicPPL,
        NamedArrays,
        Accessors
    import Turing.OptimisationCore:
        ModeResult,
        MLE,
        MAP,
        OptimLogDensity,
        variable_names,
        OptimizationContext
    import Optim
else
    import ..Turing
    import ..Turing:
        DynamicPPL,
        NamedArrays,
        Accessors
    import ..OptimisationCore:
        ModeResult,
        MLE,
        MAP,
        OptimLogDensity,
        variable_names,
        OptimizationContext
    import ..Optim
end

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
    f = Accessors.@set f.varinfo = DynamicPPL.unflatten(f.varinfo, init_vals)
    f = Accessors.@set f.varinfo = DynamicPPL.link(f.varinfo, model)
    init_vals = DynamicPPL.getparams(f)

    # Optimize!
    M = Optim.optimize(Optim.only_fg!(f), init_vals, optimizer, options, args...; kwargs...)

    # Warn the user if the optimization did not converge.
    if !Optim.converged(M)
        @warn "Optimization did not converge! You may need to correct your model or adjust the Optim parameters."
    end

    # Get the VarInfo at the MLE/MAP point, and run the model to ensure
    # correct dimensionality.
    f = Accessors.@set f.varinfo = DynamicPPL.unflatten(f.varinfo, M.minimizer)
    f = Accessors.@set f.varinfo = DynamicPPL.invlink(f.varinfo, model)
    vals = DynamicPPL.getparams(f)
    f = Accessors.@set f.varinfo = DynamicPPL.link(f.varinfo, model)

    # Store the parameters and their names in an array.
    varnames = map(Symbol ∘ first, Turing.Inference.getparams(model, f.varinfo))
    vmat = NamedArrays.NamedArray(vals, varnames)

    return Turing.ModeResult(vmat, M, -M.minimum, f)
end

end # module
