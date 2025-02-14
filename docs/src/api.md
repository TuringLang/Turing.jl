# API

## Module-wide re-exports

Turing.jl directly re-exports the entire public API of the following packages:

  - [Distributions.jl](https://juliastats.org/Distributions.jl)
  - [MCMCChains.jl](https://turinglang.org/MCMCChains.jl)
  - [AbstractMCMC.jl](https://turinglang.org/AbstractMCMC.jl)
  - [Bijectors.jl](https://turinglang.org/Bijectors.jl)
  - [Libtask.jl](https://github.com/TuringLang/Libtask.jl)

Please see the individual packages for their documentation.

## Individual exports and re-exports

**All** of the following symbols are exported unqualified by Turing, even though the documentation suggests that many of them are qualified.
That means, for example, you can just write

```julia
using Turing

@model function my_model() end

sample(my_model(), Prior(), 100)
```

instead of

```julia
DynamicPPL.@model function my_model() end

sample(my_model(), Turing.Inference.Prior(), 100)
```

even though [`Prior()`](@ref) is actually defined in the `Turing.Inference` module and [`@model`](@ref) in the `DynamicPPL` package.

### Modelling

| Exported symbol | Documentation                       | Description                                  |
|:--------------- |:----------------------------------- |:-------------------------------------------- |
| `@model`        | [`DynamicPPL.@model`](@extref)      | Define a probabilistic model                 |
| `@varname`      | [`AbstractPPL.@varname`](@extref)   | Generate a `VarName` from a Julia expression |
| `to_submodel`   | [`DynamicPPL.to_submodel`](@extref) | Define a submodel                            |

### Inference

| Exported symbol | Documentation                                                                                    | Description         |
|:--------------- |:------------------------------------------------------------------------------------------------ |:------------------- |
| `sample`        | [`StatsBase.sample`](https://turinglang.org/AbstractMCMC.jl/stable/api/#Sampling-a-single-chain) | Sample from a model |

### Samplers

| Exported symbol      | Documentation                                 | Description                                                         |
|:-------------------- |:--------------------------------------------- |:------------------------------------------------------------------- |
| `Prior`              | [`Turing.Inference.Prior`](@ref)              | Sample from the prior distribution                                  |
| `MH`                 | [`Turing.Inference.MH`](@ref)                 | Metropolis–Hastings                                                 |
| `Emcee`              | [`Turing.Inference.Emcee`](@ref)              | Affine-invariant ensemble sampler                                   |
| `ESS`                | [`Turing.Inference.ESS`](@ref)                | Elliptical slice sampling                                           |
| `Gibbs`              | [`Turing.Inference.Gibbs`](@ref)              | Gibbs sampling                                                      |
| `HMC`                | [`Turing.Inference.HMC`](@ref)                | Hamiltonian Monte Carlo                                             |
| `SGLD`               | [`Turing.Inference.SGLD`](@ref)               | Stochastic gradient Langevin dynamics                               |
| `SGHMC`              | [`Turing.Inference.SGHMC`](@ref)              | Stochastic gradient Hamiltonian Monte Carlo                         |
| `PolynomialStepsize` | [`Turing.Inference.PolynomialStepsize`](@ref) | Returns a function which generates polynomially decaying step sizes |
| `HMCDA`              | [`Turing.Inference.HMCDA`](@ref)              | Hamiltonian Monte Carlo with dual averaging                         |
| `NUTS`               | [`Turing.Inference.NUTS`](@ref)               | No-U-Turn Sampler                                                   |
| `IS`                 | [`Turing.Inference.IS`](@ref)                 | Importance sampling                                                 |
| `SMC`                | [`Turing.Inference.SMC`](@ref)                | Sequential Monte Carlo                                              |
| `PG`                 | [`Turing.Inference.PG`](@ref)                 | Particle Gibbs                                                      |
| `CSMC`               | [`Turing.Inference.CSMC`](@ref)               | The same as PG                                                      |
| `externalsampler`    | [`Turing.Inference.externalsampler`](@ref)    | Wrap an external sampler for use in Turing                          |

### Variational inference

See the [variational inference tutorial](https://turinglang.org/docs/tutorials/09-variational-inference/) for a walkthrough on how to use these.

| Exported symbol | Documentation                | Description                             |
|:--------------- |:---------------------------- |:--------------------------------------- |
| `vi`            | [`AdvancedVI.vi`](@extref)   | Perform variational inference           |
| `ADVI`          | [`AdvancedVI.ADVI`](@extref) | Construct an instance of a VI algorithm |

### Automatic differentiation types

These are used to specify the automatic differentiation backend to use.
See the [AD guide](https://turinglang.org/docs/tutorials/docs-10-using-turing-autodiff/) for more information.

| Exported symbol   | Documentation                        | Description            |
|:----------------- |:------------------------------------ |:---------------------- |
| `AutoForwardDiff` | [`ADTypes.AutoForwardDiff`](@extref) | ForwardDiff.jl backend |
| `AutoReverseDiff` | [`ADTypes.AutoReverseDiff`](@extref) | ReverseDiff.jl backend |
| `AutoZygote`      | [`ADTypes.AutoZygote`](@extref)      | Zygote.jl backend      |
| `AutoMooncake`    | [`ADTypes.AutoMooncake`](@extref)    | Mooncake.jl backend    |

### Debugging

```@docs
setprogress!
```

### Distributions

These distributions are defined in Turing.jl, but not in Distributions.jl.

```@docs
Flat
FlatPos
BinomialLogit
OrderedLogistic
LogPoisson
```

`BernoulliLogit` is part of Distributions.jl since version 0.25.77.
If you are using an older version of Distributions where this isn't defined, Turing will export the same distribution.

```@docs
Distributions.BernoulliLogit
```

### Tools to work with distributions

| Exported symbol | Documentation                          | Description                                                    |
|:--------------- |:-------------------------------------- |:-------------------------------------------------------------- |
| `filldist`      | [`DistributionsAD.filldist`](@extref)  | Create a product distribution from a distribution and integers |
| `arraydist`     | [`DistributionsAD.arraydist`](@extref) | Create a product distribution from an array of distributions   |
| `NamedDist`     | [`DynamicPPL.NamedDist`](@extref)      | A distribution that carries the name of the variable           |

### Predictions

```@docs
predict
```

### Querying model probabilities and quantities

Please see the [generated quantities](https://turinglang.org/docs/tutorials/usage-generated-quantities/) and [probability interface](https://turinglang.org/docs/tutorials/usage-probability-interface/) guides for more information.

| Exported symbol            | Documentation                                                                                                                     | Description                                                     |
|:-------------------------- |:--------------------------------------------------------------------------------------------------------------------------------- |:--------------------------------------------------------------- |
| `generated_quantities`     | [`DynamicPPL.generated_quantities`](@extref)                                                                                      | Calculate additional quantities defined in a model              |
| `pointwise_loglikelihoods` | [`DynamicPPL.pointwise_loglikelihoods`](@extref)                                                                                  | Compute log likelihoods for each sample in a chain              |
| `logprior`                 | [`DynamicPPL.logprior`](@extref)                                                                                                  | Compute log prior probability                                   |
| `logjoint`                 | [`DynamicPPL.logjoint`](@extref)                                                                                                  | Compute log joint probability                                   |
| `LogDensityFunction`       | [`DynamicPPL.LogDensityFunction`](@extref)                                                                                        | Wrap a Turing model to satisfy LogDensityFunctions.jl interface |
| `condition`                | [`AbstractPPL.condition`](@extref)                                                                                                | Condition a model on data                                       |
| `decondition`              | [`AbstractPPL.decondition`](@extref)                                                                                              | Remove conditioning on data                                     |
| `conditioned`              | [`DynamicPPL.conditioned`](@extref)                                                                                               | Return the conditioned values of a model                        |
| `fix`                      | [`DynamicPPL.fix`](@extref)                                                                                                       | Fix the value of a variable                                     |
| `unfix`                    | [`DynamicPPL.unfix`](@extref)                                                                                                     | Unfix the value of a variable                                   |
| `OrderedDict`              | [`OrderedCollections.OrderedDict`](https://juliacollections.github.io/OrderedCollections.jl/dev/ordered_containers/#OrderedDicts) | An ordered dictionary                                           |

### Extra re-exports from Bijectors

Note that Bijectors itself does not export `ordered`.

```@docs
Bijectors.ordered
```

### Point estimates

See the [mode estimation tutorial](https://turinglang.org/docs/tutorials/docs-17-mode-estimation/) for more information.

| Exported symbol        | Documentation                                      | Description                                  |
|:---------------------- |:-------------------------------------------------- |:-------------------------------------------- |
| `maximum_a_posteriori` | [`Turing.Optimisation.maximum_a_posteriori`](@ref) | Find a MAP estimate for a model              |
| `maximum_likelihood`   | [`Turing.Optimisation.maximum_likelihood`](@ref)   | Find a MLE estimate for a model              |
| `MAP`                  | [`Turing.Optimisation.MAP`](@ref)                  | Type to use with Optim.jl for MAP estimation |
| `MLE`                  | [`Turing.Optimisation.MLE`](@ref)                  | Type to use with Optim.jl for MLE estimation |
