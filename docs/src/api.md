# API

## Module-wide re-exports

Turing.jl directly re-exports the entire public API of the following packages:

  - [Distributions.jl](https://juliastats.org/Distributions.jl)
  - [MCMCChains.jl](https://turinglang.org/MCMCChains.jl)

Please see the individual packages for their documentation.

## Individual exports and re-exports

In this API documentation, for the sake of clarity, we have listed the module that actually defines each of the exported symbols.
Note, however, that **all** of the following symbols are exported unqualified by Turing.
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

even though [`Prior()`](@ref) is actually defined in the `Turing.Inference` module and [`@model`](@extref `DynamicPPL.@model`) in the `DynamicPPL` package.

### Modelling

| Exported symbol      | Documentation                              | Description                                                                                  |
|:-------------------- |:------------------------------------------ |:-------------------------------------------------------------------------------------------- |
| `@model`             | [`DynamicPPL.@model`](@extref)             | Define a probabilistic model                                                                 |
| `@varname`           | [`AbstractPPL.@varname`](@extref)          | Generate a `VarName` from a Julia expression                                                 |
| `to_submodel`        | [`DynamicPPL.to_submodel`](@extref)        | Define a submodel                                                                            |
| `prefix`             | [`DynamicPPL.prefix`](@extref)             | Prefix all variable names in a model with a given VarName                                    |
| `LogDensityFunction` | [`DynamicPPL.LogDensityFunction`](@extref) | A struct containing all information about how to evaluate a model. Mostly for advanced users |
| `@addlogprob!`       | [`DynamicPPL.@addlogprob!`](@extref)       | Add arbitrary log-probability terms during model evaluation                                  |
| `setthreadsafe`      | [`DynamicPPL.setthreadsafe`](@extref)      | Mark a model as requiring threadsafe evaluation                                              |

### Inference

| Exported symbol   | Documentation                                                             | Description                               |
|:----------------- |:------------------------------------------------------------------------- |:----------------------------------------- |
| `sample`          | [`StatsBase.sample`](https://turinglang.org/docs/usage/sampling-options/) | Sample from a model                       |
| `MCMCThreads`     | [`AbstractMCMC.MCMCThreads`](@extref)                                     | Run MCMC using multiple threads           |
| `MCMCDistributed` | [`AbstractMCMC.MCMCDistributed`](@extref)                                 | Run MCMC using multiple processes         |
| `MCMCSerial`      | [`AbstractMCMC.MCMCSerial`](@extref)                                      | Run MCMC using without parallelism        |
| `loadstate`       | [`Turing.Inference.loadstate`](@ref)                                      | Load saved state from `MCMCChains.Chains` |

### Samplers

| Exported symbol      | Documentation                                 | Description                                                         |
|:-------------------- |:--------------------------------------------- |:------------------------------------------------------------------- |
| `Prior`              | [`Turing.Inference.Prior`](@ref)              | Sample from the prior distribution                                  |
| `MH`                 | [`Turing.Inference.MH`](@ref)                 | Metropolis–Hastings                                                 |
| `Emcee`              | [`Turing.Inference.Emcee`](@ref)              | Affine-invariant ensemble sampler                                   |
| `ESS`                | [`Turing.Inference.ESS`](@ref)                | Elliptical slice sampling                                           |
| `Gibbs`              | [`Turing.Inference.Gibbs`](@ref)              | Gibbs sampling                                                      |
| `GibbsConditional`   | [`Turing.Inference.GibbsConditional`](@ref)   | Gibbs sampling with analytical conditional posterior distributions  |
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
| `RepeatSampler`      | [`Turing.Inference.RepeatSampler`](@ref)      | A sampler that runs multiple times on the same variable             |
| `externalsampler`    | [`Turing.Inference.externalsampler`](@ref)    | Wrap an external sampler for use in Turing                          |

### DynamicPPL utilities

Please see the [generated quantities](https://turinglang.org/docs/tutorials/usage-generated-quantities/) and [probability interface](https://turinglang.org/docs/tutorials/usage-probability-interface/) guides for more information.

| Exported symbol            | Documentation                                                                                                                | Description                                             |
|:-------------------------- |:---------------------------------------------------------------------------------------------------------------------------- |:------------------------------------------------------- |
| `returned`                 | [`DynamicPPL.returned`](https://turinglang.org/DynamicPPL.jl/stable/api/#DynamicPPL.returned-Tuple%7BModel,%20NamedTuple%7D) | Calculate additional quantities defined in a model      |
| `predict`                  | [`StatsAPI.predict`](https://turinglang.org/DynamicPPL.jl/stable/api/#Predicting)                                            | Generate samples from posterior predictive distribution |
| `pointwise_loglikelihoods` | [`DynamicPPL.pointwise_loglikelihoods`](@extref)                                                                             | Compute log likelihoods for each sample in a chain      |
| `logprior`                 | [`DynamicPPL.logprior`](@extref)                                                                                             | Compute log prior probability                           |
| `logjoint`                 | [`DynamicPPL.logjoint`](@extref)                                                                                             | Compute log joint probability                           |
| `condition`                | [`AbstractPPL.condition`](@extref)                                                                                           | Condition a model on data                               |
| `decondition`              | [`AbstractPPL.decondition`](@extref)                                                                                         | Remove conditioning on data                             |
| `conditioned`              | [`DynamicPPL.conditioned`](@extref)                                                                                          | Return the conditioned values of a model                |
| `fix`                      | [`DynamicPPL.fix`](@extref)                                                                                                  | Fix the value of a variable                             |
| `unfix`                    | [`DynamicPPL.unfix`](@extref)                                                                                                | Unfix the value of a variable                           |
| `OrderedDict`              | [`OrderedCollections.OrderedDict`](@extref)                                                                                  | An ordered dictionary                                   |

### Initialisation strategies

Turing.jl provides several strategies to initialise parameters for models.

| Exported symbol   | Documentation                           | Description                                                     |
|:----------------- |:--------------------------------------- |:--------------------------------------------------------------- |
| `InitFromPrior`   | [`DynamicPPL.InitFromPrior`](@extref)   | Obtain initial parameters from the prior distribution           |
| `InitFromUniform` | [`DynamicPPL.InitFromUniform`](@extref) | Obtain initial parameters by sampling uniformly in linked space |
| `InitFromParams`  | [`DynamicPPL.InitFromParams`](@extref)  | Manually specify (possibly a subset of) initial parameters      |

### Variational inference

See the [docs of AdvancedVI.jl](https://turinglang.org/AdvancedVI.jl/stable/) for detailed usage and the [variational inference tutorial](https://turinglang.org/docs/tutorials/09-variational-inference/) for a basic walkthrough.

| Exported symbol               | Documentation                                            | Description                                                                                                                                       |
|:----------------------------- |:-------------------------------------------------------- |:------------------------------------------------------------------------------------------------------------------------------------------------- |
| `vi`                          | [`Turing.vi`](@ref)                                      | Perform variational inference                                                                                                                     |
| `q_locationscale`             | [`Turing.Variational.q_locationscale`](@ref)             | Find a numerically non-degenerate initialization for a location-scale variational family                                                          |
| `q_meanfield_gaussian`        | [`Turing.Variational.q_meanfield_gaussian`](@ref)        | Find a numerically non-degenerate initialization for a mean-field Gaussian family                                                                 |
| `q_fullrank_gaussian`         | [`Turing.Variational.q_fullrank_gaussian`](@ref)         | Find a numerically non-degenerate initialization for a full-rank Gaussian family                                                                  |
| `KLMinRepGradDescent`         | [`Turing.Variational.KLMinRepGradDescent`](@ref)         | KL divergence minimization via stochastic gradient descent with the reparameterization gradient                                                   |
| `KLMinRepGradProxDescent`     | [`Turing.Variational.KLMinRepGradProxDescent`](@ref)     | KL divergence minimization via stochastic proximal gradient descent with the reparameterization gradient over location-scale variational families |
| `KLMinScoreGradDescent`       | [`Turing.Variational.KLMinScoreGradDescent`](@ref)       | KL divergence minimization via stochastic gradient descent with the score gradient                                                                |
| `KLMinWassFwdBwd`             | [`Turing.Variational.KLMinWassFwdBwd`](@ref)             | KL divergence minimization via Wasserstein proximal gradient descent                                                                              |
| `KLMinNaturalGradDescent`     | [`Turing.Variational.KLMinNaturalGradDescent`](@ref)     | KL divergence minimization via natural gradient descent                                                                                           |
| `KLMinSqrtNaturalGradDescent` | [`Turing.Variational.KLMinSqrtNaturalGradDescent`](@ref) | KL divergence minimization via natural gradient descent in the square-root parameterization                                                       |
| `FisherMinBatchMatch`         | [`Turing.Variational.FisherMinBatchMatch`](@ref)         | Covariance-weighted Fisher divergence minimization via the batch-and-match algorithm                                                              |

### Automatic differentiation types

These are used to specify the automatic differentiation backend to use.
See the [AD guide](https://turinglang.org/docs/tutorials/docs-10-using-turing-autodiff/) for more information.

| Exported symbol   | Documentation                        | Description            |
|:----------------- |:------------------------------------ |:---------------------- |
| `AutoForwardDiff` | [`ADTypes.AutoForwardDiff`](@extref) | ForwardDiff.jl backend |
| `AutoReverseDiff` | [`ADTypes.AutoReverseDiff`](@extref) | ReverseDiff.jl backend |
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

### Tools to work with distributions

| Exported symbol | Documentation                          | Description                                                    |
|:--------------- |:-------------------------------------- |:-------------------------------------------------------------- |
| `I`             | [`LinearAlgebra.I`](@extref)           | Identity matrix                                                |
| `filldist`      | [`DistributionsAD.filldist`](@extref)  | Create a product distribution from a distribution and integers |
| `arraydist`     | [`DistributionsAD.arraydist`](@extref) | Create a product distribution from an array of distributions   |
| `NamedDist`     | [`DynamicPPL.NamedDist`](@extref)      | A distribution that carries the name of the variable           |

### Point estimates

See the [mode estimation tutorial](https://turinglang.org/docs/tutorials/docs-17-mode-estimation/) for more information.

| Exported symbol        | Documentation                                      | Description                                  |
|:---------------------- |:-------------------------------------------------- |:-------------------------------------------- |
| `maximum_a_posteriori` | [`Turing.Optimisation.maximum_a_posteriori`](@ref) | Find a MAP estimate for a model              |
| `maximum_likelihood`   | [`Turing.Optimisation.maximum_likelihood`](@ref)   | Find a MLE estimate for a model              |
| `MAP`                  | [`Turing.Optimisation.MAP`](@ref)                  | Type to use with Optim.jl for MAP estimation |
| `MLE`                  | [`Turing.Optimisation.MLE`](@ref)                  | Type to use with Optim.jl for MLE estimation |
