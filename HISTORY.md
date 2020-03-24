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
