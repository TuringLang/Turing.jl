module ModeEstimation

using ..Turing
using ..Bijectors
using LinearAlgebra

import ..DynamicPPL
import Optim
import NamedArrays
import ..ForwardDiff

export MAP, MLE

"""
    ModeResult{
        V<:NamedArrays.NamedArray, 
        M<:NamedArrays.NamedArray, 
        O<:Optim.MultivariateOptimizationResults, 
        S<:NamedArrays.NamedArray
    }

A wrapper struct to store various results from a MAP or MLE estimation.

Fields:

- `values` is a vector with the resulting point estimates
- `info_matrix` is the inverse Hessian
- `optim_result` is the stored Optim.jl results
- `summary_table` is a summary table with parameters, standard errors, and 
  t-statistics computed from the information matrix.
- `lp` is the final likelihood.
"""
struct ModeResult{
    V<:NamedArrays.NamedArray, 
    M<:Union{Missing, NamedArrays.NamedArray}, 
    O<:Optim.MultivariateOptimizationResults, 
    S<:NamedArrays.NamedArray
}
    values :: V
    info_matrix :: M
    optim_result :: O
    summary_table :: S
    lp :: Float64
end

function Base.show(io::IO, m::ModeResult)
    show(io, m.summary_table)
end

"""
    make_logjoint(model::DynamicPPL.Model, ctx::DynamicPPL.AbstractContext)

Construct a log density function.

The `model` is run using the provided context `ctx`.
"""
function make_logjoint(model::DynamicPPL.Model, ctx::DynamicPPL.AbstractContext)
    # setup
    varinfo_init = Turing.VarInfo(model)
    spl = DynamicPPL.SampleFromPrior()    
    DynamicPPL.link!(varinfo_init, spl)

    function logπ(z; unlinked = false)
        varinfo = DynamicPPL.VarInfo(varinfo_init, spl, z)

        unlinked && DynamicPPL.invlink!(varinfo_init, spl)
        model(varinfo, spl, ctx)
        unlinked && DynamicPPL.link!(varinfo_init, spl)

        return -DynamicPPL.getlogp(varinfo)
    end

    return logπ
end

"""
    mode_estimation(
        model::DynamicPPL.Model, 
        lpf; 
        optim_options=Optim.Options(),
        kwargs...
    )

An internal function that handles the computation of a MLE or MAP estimate.

Arguments: 

- `model` is a `DynamicPPL.Model`.
- `lpf` is a function returned by `make_logjoint`.

Optional arguments:

- `optim_options` is a `Optim.Options` struct that allows you to change the number
  of iterations run in an MLE estimate.

"""
function mode_estimation(
    model::DynamicPPL.Model, 
    lpf; 
    optim_options=Optim.Options(),
    kwargs...
)
    # Do some initialization.
    b = bijector(model)
    binv = inv(b)

    spl = DynamicPPL.SampleFromPrior()
    vi = DynamicPPL.VarInfo(model)
    init_params = model(vi, spl)
    init_vals = vi[spl]

    # Construct target function.
    target(x) = lpf(x)
    hess_target(x) = lpf(x; unlinked=true)

    # Optimize!
    M = Optim.optimize(target, init_vals, optim_options)

    # Retrieve the estimated values.
    vals = binv(M.minimizer)

    # Get the VarInfo at the MLE/MAP point, and run the model to ensure 
    # correct dimensionality.
    vi[spl] = vals
    model(vi) # XXX: Is this a necessary step?

    # Make one transition to get the parameter names.
    ts = [Turing.Inference.Transition(DynamicPPL.tonamedtuple(vi), DynamicPPL.getlogp(vi))]
    varnames, _ = Turing.Inference._params_to_array(ts)

    # Store the parameters and their names in an array.
    vmat = NamedArrays.NamedArray(vals, varnames)

    # Try to generate the information matrix.
    try
        # Calculate Hessian and information matrix.
        info = ForwardDiff.hessian(hess_target, vals)
        info = inv(info)
        mat = NamedArrays.NamedArray(info, (varnames, varnames))

        # Create the standard errors.
        ses = sqrt.(diag(info))

        # Calculate t-stats.
        tstat = vals ./ ses

        # Make a summary table.
        stable = NamedArrays.NamedArray(
            [vals ses tstat], 
            (varnames, ["parameter", "std_err", "tstat"]))

        # Return a wrapped-up table.
        return ModeResult(vmat, mat, M, stable, M.minimum)
    catch err
        @warn "Could not compute Hessian matrix" err
        stable = NamedArrays.NamedArray([vals repeat([missing], length(vals)) repeat([missing], length(vals))], (varnames, ["parameter", "std_err", "tstat"]))
        return ModeResult(vmat, missing, M, stable, M.minimum)
    end
end

"""
    MLE(model::DynamicPPL.Model; kwargs...)

Returns a maximum likelihood estimate of the given `model`.

Arguments: 

- `model` is a `DynamicPPL.Model`.

Keyword arguments:

- `optim_options` is a `Optim.Options` struct that allows you to change the number
  of iterations run in an MLE estimate.

Usage:

```julia
using Turing

@model function f()
    m ~ Normal(0, 1)
    1.5 ~ Normal(m, 1)
    2.0 ~ Normal(m, 1)
end

model = f()
mle_estimate = MLE(model)

# Manually setting the optimizers settings.
mle_estimate = MLE(model, optim_options=Optim.Options(iterations=500))
```
"""
function MLE(model::DynamicPPL.Model; kwargs...)
    lpf = make_logjoint(model, DynamicPPL.LikelihoodContext())
    return mode_estimation(model, lpf; kwargs...)
end

"""
    MAP(model::DynamicPPL.Model; kwargs...)

Returns the maximum a posteriori estimate of the given `model`.

Arguments: 

- `model` is a `DynamicPPL.Model`.

Keyword arguments:

- `optim_options` is a `Optim.Options` struct that allows you to change the number
  of iterations run in an MLE estimate.

Usage:

```julia
using Turing

@model function f()
    m ~ Normal(0, 1)
    1.5 ~ Normal(m, 1)
    2.0 ~ Normal(m, 1)
end

model = f()
mle_estimate = MAP(model)

# Manually setting the optimizers settings.
mle_estimate = MAP(model, optim_options=Optim.Options(iterations=500))
```
"""
function MAP(model::DynamicPPL.Model; kwargs...)
    lpf = make_logjoint(model, DynamicPPL.DefaultContext())
    return mode_estimation(model, lpf; kwargs...)
end

end #module 
