# Release 0.12.0

- The interface for defining new distributions with a constrained support and making them compatible with `Turing` has changed. To make a custom distribution type `CustomDistribution` compatible with `Turing`, the user needs to define the method `bijector(d::CustomDistribution)` that returns an instance of type `Bijector` implementing the `Bijectors.Bijector` API.
- `~` is now thread-safe when used for observations, but not assumptions yet.
- There were some performance improvements in the automatic differentiation of functions in `DistributionsAD` and `Bijectors` leading to speeds closer to and sometimes faster than Stan's.
- An HMC initialization bug was fixed. HMC initialization in Turing is now consistent with Stan's.
- Sampling from the prior is now possible using `sample`.
- `psample` is now deprecated, in favor of `sample(model, sampler, parallel_method, n_samples, n_chains)` where `parallel_method` can be either `MCMCThreads()` or `MCMCDistributed()`. `MCMCThreads` will use your available threads to sample each chain (ensure that you have the environment variable `JULIA_NUM_THREADS` set to the number of threads you want to use) and `MCMCDistributed` will dispatch chain sampling to each available processes (you can add processes with `addprocs()`).
- Turing now uses AdvancedMH v0.5, which mostly provides behind-the -scenes restructuring.
- Custom expressions and macros can be interpolated in the `@model` definition with `$` and it is possible to use `@.` also for assumptions and observations.
- The macros `@varinfo`, `@logpdf`, and `@sampler` are removed. Instead one can access the internal variables `_varinfo`, `_model`, `_sampler`, and `_context` in the `@model` definition.
- Additional constructors for SMC and PG make it easier to choose the resampling method and threshold.

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
- Turing now has experimental support for Zygote.jl ([#783](https://github.com/TuringLang/Turing.jl/pull/783)) and ReverseDiff.jl ([#1170](https://github.com/TuringLang/Turing.jl/pull/1170)) AD backends. Both backends are experimental, so please report any bugs you find. Zygote does not allow mutation within your model, so please be aware of this issue. You can enable Zygote with `Turing.setadbackend(:zygote)` and you can enable ReverseDiff with `Turing.setadbackend(:reversediff)`, though to use either you must import the package with `using Zygote` or `using ReverseDiff`. `for` loops are not recommended for ReverseDiff or Zygote -- see [performance tips](https://turing.ml/dev/docs/using-turing/performancetips#special-care-for-tracker-and-zygote) for more information. 
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
-  NUTS implementation #188
-  HMC: Transforms of ϵ for each variable #67 (replace with introducing mass matrix)
-  Finish: Sampler (internal) interface design #107
-  Substantially improve performance of HMC and Gibbs #7 
  -  Vectorising likelihood computations #117 #255
 -  Remove obsolete `randoc`, `randc`? #156
-  Support truncated distribution. #87
-  Refactoring code: Unify VarInfo, Trace, TaskLocalStorage #96
-  Refactoring code: Better gradient interface #97

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
