# Release 0.39.3

Improved the performance of `Turing.Inference.getparams` when called with an untyped VarInfo as the second argument, by first converting to a typed VarInfo.
This makes, for example, the post-sampling Chains construction for `Prior()` run much faster.

# Release 0.39.2

Fixed a bug in the support of `OrderedLogistic` (by changing the minimum from 0 to 1).

# Release 0.39.1

No changes from 0.39.0 — this patch is released just to re-trigger a Documenter.jl run.

# Release 0.39.0

## Update to the AdvancedVI interface

Turing's variational inference interface was updated to match version 0.4 version of AdvancedVI.jl.

AdvancedVI v0.4 introduces various new features:

  - location-scale families with dense scale matrices,
  - parameter-free stochastic optimization algorithms like `DoG` and `DoWG`,
  - proximal operators for stable optimization,
  - the sticking-the-landing control variate for faster convergence, and
  - the score gradient estimator for non-differentiable targets.

Please see the [Turing API documentation](https://turinglang.org/Turing.jl/stable/api/#Variational-inference), and [AdvancedVI's documentation](https://turinglang.org/AdvancedVI.jl/stable/), for more details.

## Removal of Turing.Essential

The Turing.Essential module has been removed.
Anything exported from there can be imported from either `Turing` or `DynamicPPL`.

## `@addlogprob!`

The `@addlogprob!` macro is now exported from Turing, making it officially part of the public interface.

# Release 0.38.6

Added compatibility with AdvancedHMC 0.8.

# Release 0.38.5

Added compatibility with ForwardDiff v1.

# Release 0.38.4

The minimum Julia version was increased to 1.10.2 (from 1.10.0).
On versions before 1.10.2, `sample()` took an excessively long time to run (probably due to compilation).

# Release 0.38.3

`getparams(::Model, ::AbstractVarInfo)` now returns an empty `Float64[]` if the VarInfo contains no parameters.

# Release 0.38.2

Bump compat for `MCMCChains` to `7`.
By default, summary statistics and quantiles for chains are no longer printed; to access these you should use `describe(chain)`.

# Release 0.38.1

The method `Bijectors.bijector(::DynamicPPL.Model)` was moved to DynamicPPL.jl.

# Release 0.38.0

## DynamicPPL version

DynamicPPL compatibility has been bumped to 0.36.
This brings with it a number of changes: the ones most likely to affect you are submodel prefixing and conditioning.
Variables in submodels are now represented correctly with field accessors.
For example:

```julia
using Turing
@model inner() = x ~ Normal()
@model outer() = a ~ to_submodel(inner())
```

`keys(VarInfo(outer()))` now returns `[@varname(a.x)]` instead of `[@varname(var"a.x")]`

Furthermore, you can now either condition on the outer model like `outer() | (@varname(a.x) => 1.0)`, or the inner model like `inner() | (@varname(x) => 1.0)`.
If you use the conditioned inner model as a submodel, the conditioning will still apply correctly.

Please see [the DynamicPPL release notes](https://github.com/TuringLang/DynamicPPL.jl/releases/tag/v0.36.0) for fuller details.

## Gibbs sampler

Turing's Gibbs sampler now allows for more complex `VarName`s, such as `x[1]` or `x.a`, to be used.
For example, you can now do this:

```julia
@model function f()
    x = Vector{Float64}(undef, 2)
    x[1] ~ Normal()
    return x[2] ~ Normal()
end
sample(f(), Gibbs(@varname(x[1]) => MH(), @varname(x[2]) => MH()), 100)
```

Performance for the cases which used to previously work (i.e. `VarName`s like `x` which only consist of a single symbol) is unaffected, and `VarNames` with only field accessors (e.g. `x.a`) should be equally fast.
It is possible that `VarNames` with indexing (e.g. `x[1]`) may be slower (although this is still an improvement over not working at all!).
If you find any cases where you think the performance is worse than it should be, please do file an issue.

# Release 0.37.1

`maximum_a_posteriori` and `maximum_likelihood` now perform sanity checks on the model before running the optimisation.
To disable this, set the keyword argument `check_model=false`.

# Release 0.37.0

## Breaking changes

### Gibbs constructors

0.37 removes the old Gibbs constructors deprecated in 0.36.

### Remove Zygote support

Zygote is no longer officially supported as an automatic differentiation backend, and `AutoZygote` is no longer exported. You can continue to use Zygote by importing `AutoZygote` from ADTypes and it may well continue to work, but it is no longer tested and no effort will be expended to fix it if something breaks.

[Mooncake](https://github.com/compintell/Mooncake.jl/) is the recommended replacement for Zygote.

### DynamicPPL 0.35

Turing.jl v0.37 uses DynamicPPL v0.35, which brings with it several breaking changes:

  - The right hand side of `.~` must from now on be a univariate distribution.
  - Indexing `VarInfo` objects by samplers has been removed completely.
  - The order in which nested submodel prefixes are applied has been reversed.
  - The arguments for the constructor of `LogDensityFunction` have changed. `LogDensityFunction` also now satisfies the `LogDensityProblems` interface, without needing a wrapper object.

For more details about all of the above, see the changelog of DynamicPPL [here](https://github.com/TuringLang/DynamicPPL.jl/releases/tag/v0.35.0).

### Export list

Turing.jl's export list has been cleaned up a fair bit. This affects what is imported into your namespace when you do an unqualified `using Turing`. You may need to import things more explicitly than before.

  - The `DynamicPPL` and `AbstractMCMC` modules are no longer exported. You will need to `import DynamicPPL` or `using DynamicPPL: DynamicPPL` (likewise `AbstractMCMC`) yourself, which in turn means that they have to be made available in your project environment.

  - `@logprob_str` and `@prob_str` have been removed following a long deprecation period.
  - We no longer re-export everything from `Bijectors` and `Libtask`. To get around this, add `using Bijectors` or `using Libtask` at the top of your script (but we recommend using more selective imports).
    
      + We no longer export `Bijectors.ordered`. If you were using `ordered`, even Bijectors does not (currently) export this. You will have to manually import it with `using Bijectors: ordered`.

On the other hand, we have added a few more exports:

  - `DynamicPPL.returned` and `DynamicPPL.prefix` are exported (for use with submodels).
  - `LinearAlgebra.I` is exported for convenience.

# Release 0.36.0

## Breaking changes

0.36.0 introduces a new Gibbs sampler. It's been included in several previous releases as `Turing.Experimental.Gibbs`, but now takes over the old Gibbs sampler, which gets removed completely.

The new Gibbs sampler currently supports the same user-facing interface as the old one, but the old constructors have been deprecated, and will be removed in the future. Also, given that the internals have been completely rewritten in a very different manner, there may be accidental breakage that we haven't anticipated. Please report any you find.

`GibbsConditional` has also been removed. It was never very user-facing, but it was exported, so technically this is breaking.

The old Gibbs constructor relied on being called with several subsamplers, and each of the constructors of the subsamplers would take as arguments the symbols for the variables that they are to sample, e.g. `Gibbs(HMC(:x), MH(:y))`. This constructor has been deprecated, and will be removed in the future. The new constructor works by mapping symbols, `VarName`s, or iterables thereof to samplers, e.g. `Gibbs(x=>HMC(), y=>MH())`, `Gibbs(@varname(x) => HMC(), @varname(y) => MH())`, `Gibbs((:x, :y) => NUTS(), :z => MH())`. This allows more granular specification of which sampler to use for which variable.

Likewise, the old constructor for calling one subsampler more often than another, `Gibbs((HMC(0.01, 4, :x), 2), (MH(:y), 1))` has been deprecated. The new way to do this is to use `RepeatSampler`, also introduced at this version: `Gibbs(@varname(x) => RepeatSampler(HMC(0.01, 4), 2), @varname(y) => MH())`.

# Release 0.35.0

## Breaking changes

Julia 1.10 is now the minimum required version for Turing.

Tapir.jl has been removed and replaced with its successor, Mooncake.jl.
You can use Mooncake.jl by passing `adbackend=AutoMooncake(; config=nothing)` to the relevant samplers.

Support for Tracker.jl as an AD backend has been removed.

# Release 0.33.0

## Breaking changes

The following exported functions have been removed:

  - `constrained_space`
  - `get_parameter_bounds`
  - `optim_objective`
  - `optim_function`
  - `optim_problem`

The same functionality is now offered by the new exported functions

  - `maximum_likelihood`
  - `maximum_a_posteriori`

# Release 0.30.5

  - `essential/ad.jl` is removed, `ForwardDiff` and `ReverseDiff` integrations via `LogDensityProblemsAD` are moved to `DynamicPPL` and live in corresponding package extensions.
  - `LogDensityProblemsAD.ADgradient(ℓ::DynamicPPL.LogDensityFunction)` (i.e. the single argument method) is moved to `Inference` module. It will create `ADgradient` using the `adtype` information stored in `context` field of `ℓ`.
  - `getADbackend` function is renamed to `getADType`, the interface is preserved, but packages that previously used `getADbackend` should be updated to use `getADType`.
  - `TuringTag` for ForwardDiff is also removed, now `DynamicPPLTag` is defined in `DynamicPPL` package and should serve the same [purpose](https://www.stochasticlifestyle.com/improved-forwarddiff-jl-stacktraces-with-package-tags/).

# Release 0.30.0

  - [`ADTypes.jl`](https://github.com/SciML/ADTypes.jl) replaced Turing's global AD backend. Users should now specify the desired `ADType` directly in sampler constructors, e.g., `HMC(0.1, 10; adtype=AutoForwardDiff(; chunksize))`, or `HMC(0.1, 10; adtype=AutoReverseDiff(false))` (`false` indicates not to use compiled tape).
  - Interface functions such as `ADBackend`, `setadbackend`, `setadsafe`, `setchunksize`, and `setrdcache` are deprecated and will be removed in a future release.
  - Removed the outdated `verifygrad` function.
  - Updated to a newer version of `LogDensityProblemsAD` (v1.7).

# Release 0.12.0

  - The interface for defining new distributions with constrained support and making them compatible with `Turing` has changed. To make a custom distribution type `CustomDistribution` compatible with `Turing`, the user needs to define the method `bijector(d::CustomDistribution)` that returns an instance of type `Bijector` implementing the `Bijectors.Bijector` API.
  - `~` is now thread-safe when used for observations, but not assumptions (non-observed model parameters) yet.
  - There were some performance improvements in the automatic differentiation (AD) of functions in `DistributionsAD` and `Bijectors`, leading to speeds closer to and sometimes faster than Stan's.
  - An `HMC` initialization bug was fixed. `HMC` initialization in Turing is now consistent with Stan's.
  - Sampling from the prior is now possible using `sample`.
  - `psample` is now deprecated in favour of `sample(model, sampler, parallel_method, n_samples, n_chains)` where `parallel_method` can be either `MCMCThreads()` or `MCMCDistributed()`. `MCMCThreads` will use your available threads to sample each chain (ensure that you have the environment variable `JULIA_NUM_THREADS` set to the number of threads you want to use), and `MCMCDistributed` will dispatch chain sampling to each available process (you can add processes with `addprocs()`).
  - Turing now uses `AdvancedMH.jl` v0.5, which mostly provides behind-the-scenes restructuring.
  - Custom expressions and macros can be interpolated in the `@model` definition with `$`; it is possible to use `@.` also for assumptions (non-observed model parameters) and observations.
  - The macros `@varinfo`, `@logpdf`, and `@sampler` are removed. Instead, one can access the internal variables `_varinfo`, `_model`, `_sampler`, and `_context` in the `@model` definition.
  - Additional constructors for `SMC` and `PG` make it easier to choose the resampling method and threshold.

# Release 0.11.0

  - Removed some extraneous imports and dependencies ([#1182](https://github.com/TuringLang/Turing.jl/pull/1182))
  - Minor backend changes to `sample` and `psample`, which now use functions defined upstream in AbstractMCMC.jl ([#1187](https://github.com/TuringLang/Turing.jl/pull/1187))
  - Fix for an AD-related crash ([#1202](https://github.com/TuringLang/Turing.jl/pull/1202))
  - StatsBase compat update to 0.33 ([#1185](https://github.com/TuringLang/Turing.jl/pull/1185))
  - Bugfix for ReverseDiff caching and memoization ([#1208](https://github.com/TuringLang/Turing.jl/pull/1208))
  - BREAKING: `VecBinomialLogit` is now removed. Also `BernoulliLogit` is added ([#1214](https://github.com/TuringLang/Turing.jl/pull/1214))
  - Bugfix for cases where dynamic models were breaking with HMC methods ([#1217](https://github.com/TuringLang/Turing.jl/pull/1217))
  - Updates to allow AdvancedHMC 0.2.23 ([#1218](https://github.com/TuringLang/Turing.jl/pull/1218))
  - Add more informative error messages for SMC ([#900](https://github.com/TuringLang/Turing.jl/pull/900))

# Release 0.10.1

  - Fix bug where arrays with mixed integers, floats, and missing values were not being passed to the `MCMCChains.Chains` constructor properly [#1180](https://github.com/TuringLang/Turing.jl/pull/1180).

# Release 0.10.0

  - Update elliptical slice sampling to use [EllipticalSliceSampling.jl](https://github.com/TuringLang/EllipticalSliceSampling.jl) on the backend. [#1145](https://github.com/TuringLang/Turing.jl/pull/1145). Nothing should change from a front-end perspective -- you can still call `sample(model, ESS(), 1000)`.
  - Added default progress loggers in [#1149](https://github.com/TuringLang/Turing.jl/pull/1149).
  - The symbols used to define the AD backend have changed to be the lowercase form of the package name used for AD. `forward_diff` is now `forwarddiff`, `reverse_diff` is now `tracker`, and `zygote` and `reversediff` are newly supported (see below). `forward_diff` and `reverse_diff` are deprecated and are slated to be removed.
  - Turing now has experimental support for Zygote.jl ([#783](https://github.com/TuringLang/Turing.jl/pull/783)) and ReverseDiff.jl ([#1170](https://github.com/TuringLang/Turing.jl/pull/1170)) AD backends. Both backends are experimental, so please report any bugs you find. Zygote does not allow mutation within your model, so please be aware of this issue. You can enable Zygote with `Turing.setadbackend(:zygote)` and you can enable ReverseDiff with `Turing.setadbackend(:reversediff)`, though to use either you must import the package with `using Zygote` or `using ReverseDiff`. `for` loops are not recommended for ReverseDiff or Zygote -- see [performance tips](https://turinglang.org/dev/docs/using-turing/performancetips#special-care-for-codetrackercode-and-codezygotecode) for more information.
  - Fix MH indexing bug [#1135](https://github.com/TuringLang/Turing.jl/pull/1135).
  - Fix MH array sampling [#1167](https://github.com/TuringLang/Turing.jl/pull/1167).
  - Fix bug in VI where the bijectors where being inverted incorrectly [#1168](https://github.com/TuringLang/Turing.jl/pull/1168).
  - The Gibbs sampler handles state better by passing `Transition` structs to the local samplers ([#1169](https://github.com/TuringLang/Turing.jl/pull/1169) and [#1166](https://github.com/TuringLang/Turing.jl/pull/1166)).

# Release 0.4.0-alpha

  - Fix compatibility with Julia 0.6 [#341, #330, #293]
  - Support of Stan interface [#343, #326]
  - Fix Binomial distribution for gradients. [#311]
  - Stochastic gradient Hamiltonian Monte Carlo [#201]; Stochastic gradient Langevin dynamics [#27]
  - More particle MCMC family samplers: PIMH & PMMH [#364, #369]
  - Disable adaptive resampling for CSMC [#357]
  - Fix resampler for SMC [#338]
  - Interactive particle MCMC [#334]
  - Add type alias CSMC for PG [#333]
  - Fix progress meter [#317]

# Release 0.3

  - NUTS implementation #188
  - HMC: Transforms of ϵ for each variable #67 (replace with introducing mass matrix)
  - Finish: Sampler (internal) interface design #107
  - Substantially improve performance of HMC and Gibbs #7
  - Vectorising likelihood computations #117 #255
  - Remove obsolete `randoc`, `randc`? #156
  - Support truncated distribution. #87
  - Refactoring code: Unify VarInfo, Trace, TaskLocalStorage #96
  - Refactoring code: Better gradient interface #97

# Release 0.2

  - Gibbs sampler ([#73])
  - HMC for constrained variables ([#66]; no support for varying dimensions)
  - Added support for `Mamba.Chain` ([#90]): describe, plot etc.
  - New interface design ([#55]), ([#104])
  - Bugfixes and general improvements (e.g. `VarInfo` [#96])

# Release 0.1.0

  - Initial support for Hamiltonian Monte Carlo (no support for discrete/constrained variables)
  - Require Julia 0.5
  - Bugfixes and general improvements

# Release 0.0.1-0.0.4

The initial releases of Turing.

  - Particle MCMC, SMC, IS
  - Implemented [copying for Julia Task](https://github.com/JuliaLang/julia/pull/15078)
  - Implemented copy-on-write data structure `TArray` for Tasks
