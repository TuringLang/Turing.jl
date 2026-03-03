# Mode Estimation

After defining a statistical model, in addition to sampling from its distributions, one may be interested in finding the parameter values that maximise (for instance) the posterior density, or the likelihood.
This is called mode estimation.

Turing provides support for two mode estimation techniques, [maximum likelihood estimation](https://en.wikipedia.org/wiki/Maximum_likelihood_estimation) (MLE) and [maximum a posteriori](https://en.wikipedia.org/wiki/Maximum_a_posteriori_estimation) (MAP) estimation.

We begin by defining a simple model to work with:

```@example 1
using Turing

@model function normal_model(y)
    x ~ Normal()
    y ~ Normal(x)
    return nothing
end
```

Once the model is defined, we can construct a model instance as we normally would:

```@example 1
model = normal_model(2.0)
```

In its simplest form, finding the maximum a posteriori or maximum likelihood parameters is just a function call:

```@example 1
# Generate a MLE estimate.
mle_estimate = maximum_likelihood(model)
```

```@example 1
# Generate a MAP estimate.
map_estimate = maximum_a_posteriori(model)
```

The estimates are returned as instances of the `ModeResult` type.
It has the fields `params` (a `VarNamedTuple` mapping `VarName`s to the parameter values found) and `lp` for the log probability at the optimum.
For more information, please see the docstring of `ModeResult`.

You can access individual parameter values by indexing into the `params` field with `VarName`s:

```@example 1
map_estimate.params[@varname(x)]
```

If you need a vectorised form of the parameters, you can use `vector_names_and_params`, which return a tuple of two vectors: one of `VarName`s and one of the corresponding parameter values.
(Note that these values are *always* returned in untransformed space.)

```@example 1
vector_names_and_params(map_estimate)
```

The `optim_result` field (which is not printed by default) contains the original result from the underlying optimisation solver, which is useful for diagnosing convergence issues and accessing solver-specific information:

```@example 1
map_estimate.optim_result
```

## Controlling the optimisation process

### Solvers

Under the hood, `maximum_likelihood` and `maximum_a_posteriori` use the [Optimization.jl](https://github.com/SciML/Optimization.jl) package, which provides a unified interface to many other optimisation packages.
By default Turing uses the [LBFGS](https://en.wikipedia.org/wiki/Limited-memory_BFGS) method from [Optim.jl](https://docs.sciml.ai/Optimization/stable/optimization_packages/optim/) to find the mode estimate, but we can change that to any other solver by passing it as the second argument:

```@example 1
using OptimizationOptimJL: NelderMead

maximum_likelihood(model, NelderMead())
```

Optimization.jl supports [many more solvers](https://docs.sciml.ai/Optimization/stable/); please see its documentation for details.

### Initial parameters

We can help the optimisation by giving it a starting point we know is close to the final solution.
Initial parameters are specified using `InitFromParams`, and must be provided in model space (i.e. untransformed):

```@example 1
params = VarNamedTuple(; x=0.5)
maximum_likelihood(model; initial_params=InitFromParams(params))
```

The default initialisation strategy is `InitFromPrior()`, which draws initial values from the prior.

### AD backend

You can also specify an automatic differentiation method using the `adtype` keyword argument:

```@example 1
import Mooncake

maximum_likelihood(model; adtype=AutoMooncake())
```

### Linked vs unlinked optimisation

By default, Turing transforms model parameters to an unconstrained space before optimising (`link=true`).
There are two reasons why one might want to do this:

 1. This avoids discontinuities where the log-density drops to `-Inf` outside the support of a distribution.
 2. But more importantly, this avoids situations where the original sample contains values that depend on each other.
    For example, in a `Dirichlet` distribution, the parameters must sum to 1.
    That means that if we do not perform linking, these parameters cannot be varied completely independently, which can lead to numerical issues.
    In contrast, when linking is performed, the parameters are transformed into a (shorter) vector of parameters that are completely unconstrained and independent.

Note that the parameter values returned are always in the original (untransformed) space, regardless of the `link` setting.

::: {.callout-note}

## What does 'unconstrained' really mean?

Note that the transformation to unconstrained space refers to the support of the *original* distribution prior to any optimisation constraints being applied.
For example, a parameter `x ~ Beta(2, 2)` will be transformed from the original space of `(0, 1)` to the unconstrained space of `(-Inf, Inf)` (via the logit transform).
However, it is possible that the optimisation still proceeds in a constrained space, if constraints on the parameter are specified via `lb` or `ub`.
For example, if we specify `lb=0.0` and `ub=0.2` for the same parameter, then the optimisation will proceed in the constrained space of `(-Inf, logit(0.2))`.
:::

If you want to optimise in the original parameter space instead, set `link=false`.

```@example 1
maximum_a_posteriori(model; link=false)
```

This is usually only useful under very specific circumstances, namely when your model contains distributions for which the mapping from model space to unconstrained space is dependent on another parameter's value.

### Box constraints

You can provide lower and upper bounds on parameters using the `lb` and `ub` keywords respectively.
Bounds are specified as a `VarNamedTuple` and, just like initial values, must be provided in model space (i.e. untransformed):

```@example 1
lb = VarNamedTuple(; x=0.0)
ub = VarNamedTuple(; x=0.2)
maximum_likelihood(model; lb=lb, ub=ub)
```

Turing will internally translate these bounds to unconstrained space if `link=true`; as a user you should not need to worry at all about the details of this transformation.

In this case we only have one parameter, but if there are multiple parameters and you only want to constrain some of them, you can provide bounds for the parameters you want to constrain and omit the others.

Note that for some distributions (e.g. `Dirichlet`, `LKJCholesky`), the mapping from model-space bounds to linked-space bounds is not well-defined.
In these cases, Turing will raise an error.
If you need constrained optimisation for such variables, either set `link=false` or use `LogDensityFunction` with Optimization.jl directly.

::: {.callout-note}
Generic (non-box) constraints are not supported by Turing's optimisation interface.
For these, please use `LogDensityFunction` and Optimization.jl directly.
:::

### Solver options

Any extra keyword arguments are passed through to `Optimization.solve`.
Some commonly useful ones are `maxiters`, `abstol`, and `reltol`:

```@example 1
params = VarNamedTuple(; x=-4.0)
badly_converged_mle = maximum_likelihood(
    model, NelderMead(); initial_params=InitFromParams(params), maxiters=10, reltol=1e-9
)
```

### Reproducibility

To get reproducible results, pass an `rng` as the first argument:

```@example 1
using Random: Xoshiro
maximum_a_posteriori(Xoshiro(468), model)
```

This controls the random number generator used for parameter initialisation; the actual optimisation process is deterministic.

For more details and a full list of keyword arguments, see the docstring of `Turing.Optimisation.estimate_mode`.

## Analysing your mode estimate

Turing extends several methods from `StatsBase` that can be used to analyse your mode estimation results.
Methods implemented include `vcov`, `informationmatrix`, `coeftable`, `coef`, and `coefnames`.

For example, let's examine our MLE estimate from above using `coeftable`:

```@example 1
using StatsBase: coeftable
coeftable(mle_estimate)
```

Standard errors are calculated from the Fisher information matrix (inverse Hessian of the log likelihood or log joint).
Note that standard errors calculated in this way may not always be appropriate for MAP estimates, so please be cautious in interpreting them.

The Hessian is computed using automatic differentiation.
By default, `ForwardDiff` is used, but if you are feeling brave you can specify a different backend via the `adtype` keyword argument to `informationmatrix`.
(Note that AD backend support for second-order derivatives is more limited than for first-order derivatives, so not all backends will work here.)

```@example 1
using StatsBase: informationmatrix
import ReverseDiff

informationmatrix(mle_estimate; adtype=AutoReverseDiff())
```

## Sampling with the MAP/MLE as initial states

You can begin sampling your chain from an MLE/MAP estimate by wrapping it in `InitFromParams` and providing it to the `sample` function with the keyword `initial_params`.
For example, here is how to sample from the full posterior using the MAP estimate as the starting point:

```@example 1
map_estimate = maximum_a_posteriori(model)
chain = sample(model, NUTS(), 1_000; initial_params=InitFromParams(map_estimate))
```
