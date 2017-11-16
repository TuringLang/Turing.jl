# Release 0.3.1
- Fix compibility with Julia 0.6 [#341, #330, #293]
- Support of Stan interface [#343, #326]
- Fix Binomal distribution for gradients. [#311]
- Stochastic gradient Hamiltonian Monte Carlo [#201]
- Disable adaptive resampling for CSMC [#357]
- Fix resampler for SMC [#338]
- Initial implementation of Interactive PMCMC [#334]
- Add typealias CSMC for PG [#333]
- Fix progress meter [#317]

# Release 0.2
- Gibbs sampler ([#73])
- HMC for constrained variables ([#66]; no support for varying dimensions)
- Added support for `Mamba.Chain` ([#90]): describe, plot etc.
- New interface degign ([#55]), ([#104])
- Bugfixes and general improvements (e.g. `VarInfo` [#96]) 

# Release 0.1.0
- Initial support for Hamiltonian Monte Carlo (no support for discrete/constrained variables)
- Require Julia 0.5
- Bugfixes and general improvements

# Release 0.0.1-0.0.4
The initial releases of Turing. 
- Particle MCMC, SMC, IS
- Implemented copying for Julia Task. 
