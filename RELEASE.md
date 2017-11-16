
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
