module TuringOptimExt

using Turing: Turing
using AbstractPPL: AbstractPPL
import Turing: DynamicPPL, NamedArrays, Accessors, Optimisation
using Optim: Optim

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
function Optim.optimize(
    model::DynamicPPL.Model,
    ::Optimisation.MLE,
    options::Optim.Options=Optim.Options();
    kwargs...,
)
    f = Optimisation.OptimLogDensity(model, DynamicPPL.getloglikelihood)
    init_vals = DynamicPPL.getparams(f.ldf)
    optimizer = Optim.LBFGS()
    return _mle_optimize(model, init_vals, optimizer, options; kwargs...)
end
function Optim.optimize(
    model::DynamicPPL.Model,
    ::Optimisation.MLE,
    init_vals::AbstractArray,
    options::Optim.Options=Optim.Options();
    kwargs...,
)
    optimizer = Optim.LBFGS()
    return _mle_optimize(model, init_vals, optimizer, options; kwargs...)
end
function Optim.optimize(
    model::DynamicPPL.Model,
    ::Optimisation.MLE,
    optimizer::Optim.AbstractOptimizer,
    options::Optim.Options=Optim.Options();
    kwargs...,
)
    f = Optimisation.OptimLogDensity(model, DynamicPPL.getloglikelihood)
    init_vals = DynamicPPL.getparams(f.ldf)
    return _mle_optimize(model, init_vals, optimizer, options; kwargs...)
end
function Optim.optimize(
    model::DynamicPPL.Model,
    ::Optimisation.MLE,
    init_vals::AbstractArray,
    optimizer::Optim.AbstractOptimizer,
    options::Optim.Options=Optim.Options();
    kwargs...,
)
    return _mle_optimize(model, init_vals, optimizer, options; kwargs...)
end

function _mle_optimize(model::DynamicPPL.Model, args...; kwargs...)
    f = Optimisation.OptimLogDensity(model, DynamicPPL.getloglikelihood)
    return _optimize(f, args...; kwargs...)
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
function Optim.optimize(
    model::DynamicPPL.Model,
    ::Optimisation.MAP,
    options::Optim.Options=Optim.Options();
    kwargs...,
)
    f = Optimisation.OptimLogDensity(model, DynamicPPL.getlogjoint)
    init_vals = DynamicPPL.getparams(f.ldf)
    optimizer = Optim.LBFGS()
    return _map_optimize(model, init_vals, optimizer, options; kwargs...)
end
function Optim.optimize(
    model::DynamicPPL.Model,
    ::Optimisation.MAP,
    init_vals::AbstractArray,
    options::Optim.Options=Optim.Options();
    kwargs...,
)
    optimizer = Optim.LBFGS()
    return _map_optimize(model, init_vals, optimizer, options; kwargs...)
end
function Optim.optimize(
    model::DynamicPPL.Model,
    ::Optimisation.MAP,
    optimizer::Optim.AbstractOptimizer,
    options::Optim.Options=Optim.Options();
    kwargs...,
)
    f = Optimisation.OptimLogDensity(model, DynamicPPL.getlogjoint)
    init_vals = DynamicPPL.getparams(f.ldf)
    return _map_optimize(model, init_vals, optimizer, options; kwargs...)
end
function Optim.optimize(
    model::DynamicPPL.Model,
    ::Optimisation.MAP,
    init_vals::AbstractArray,
    optimizer::Optim.AbstractOptimizer,
    options::Optim.Options=Optim.Options();
    kwargs...,
)
    return _map_optimize(model, init_vals, optimizer, options; kwargs...)
end

function _map_optimize(model::DynamicPPL.Model, args...; kwargs...)
    f = Optimisation.OptimLogDensity(model, DynamicPPL.getlogjoint)
    return _optimize(f, args...; kwargs...)
end

"""
    _optimize(f::OptimLogDensity, optimizer=Optim.LBFGS(), args...; kwargs...)

Estimate a mode, i.e., compute a MLE or MAP estimate.
"""
function _optimize(
    f::Optimisation.OptimLogDensity,
    init_vals::AbstractArray=DynamicPPL.getparams(f.ldf),
    optimizer::Optim.AbstractOptimizer=Optim.LBFGS(),
    options::Optim.Options=Optim.Options(),
    args...;
    kwargs...,
)
    # Convert the initial values, since it is assumed that users provide them
    # in the constrained space.
    # TODO(penelopeysm): As with in src/optimisation/Optimisation.jl, unclear
    # whether initialisation is really necessary at all
    vi = DynamicPPL.unflatten(f.ldf.varinfo, init_vals)
    vi = DynamicPPL.link(vi, f.ldf.model)
    f = Optimisation.OptimLogDensity(
        f.ldf.model, f.ldf.getlogdensity, vi; adtype=f.ldf.adtype
    )
    init_vals = DynamicPPL.getparams(f.ldf)

    # Optimize!
    M = Optim.optimize(Optim.only_fg!(f), init_vals, optimizer, options, args...; kwargs...)

    # Warn the user if the optimization did not converge.
    if !Optim.converged(M)
        @warn """
            Optimization did not converge! You may need to correct your model or adjust the
            Optim parameters.
        """
    end

    # Get the optimum in unconstrained space. `getparams` does the invlinking.
    vi = f.ldf.varinfo
    vi_optimum = DynamicPPL.unflatten(vi, M.minimizer)
    logdensity_optimum = Optimisation.OptimLogDensity(
        f.ldf.model, f.ldf.getlogdensity, vi_optimum; adtype=f.ldf.adtype
    )
    vals_dict = Turing.Inference.getparams(f.ldf.model, vi_optimum)
    iters = map(AbstractPPL.varname_and_value_leaves, keys(vals_dict), values(vals_dict))
    vns_vals_iter = mapreduce(collect, vcat, iters)
    varnames = map(Symbol ∘ first, vns_vals_iter)
    vals = map(last, vns_vals_iter)
    vmat = NamedArrays.NamedArray(vals, varnames)
    return Optimisation.ModeResult(vmat, M, -M.minimum, logdensity_optimum, vals_dict)
end

end # module
