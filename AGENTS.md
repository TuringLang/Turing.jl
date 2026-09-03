# AGENTS.md

This file provides guidance to coding agents working in this repository.

## Project Overview

Turing.jl is the user-facing entry point for the [TuringLang](https://github.com/TuringLang) probabilistic programming ecosystem. It is largely a translation layer between DynamicPPL models — which work with named, structured parameters — and inference algorithms that expect flat, vectorised samples (e.g. HMC/NUTS operate on `AbstractVector{<:Real}`). DynamicPPL's `LogDensityFunction` handles most of this translation; Turing provides the sampler wrappers that set it up and manage state across iterations.

Model definition lives in [DynamicPPL.jl](https://github.com/TuringLang/DynamicPPL.jl), parameter transformations in [Bijectors.jl](https://github.com/TuringLang/Bijectors.jl), and sampling interfaces in [AbstractMCMC.jl](https://github.com/TuringLang/AbstractMCMC.jl). Turing re-exports their APIs and provides concrete sampler implementations that wire everything together.

## Building and Testing

Code formatting uses [JuliaFormatter.jl](https://github.com/domluna/JuliaFormatter.jl) v1 (not v2) with the **Blue style** (configured in `.JuliaFormatter.toml`). CI enforces formatting on all PRs. JuliaFormatter must be installed in the **global** Julia environment, not the project environment — do not use `--project`. See the [formatting guide](https://turinglang.org/docs/contributing/code-formatting/) for setup details.

```bash
julia -e 'using JuliaFormatter; format(".")'
```

Tests use `SelectiveTests.jl` (in `test/test_utils/`) to filter by path. CI splits the suite into four shards: `mcmc/gibbs.jl`, `mcmc/Inference.jl`, `ad.jl`, and everything else. To run a subset locally:

```bash
julia --project -e 'using Pkg; Pkg.test(; test_args=["mcmc/hmc.jl"])'
```

Use `--skip` to exclude files:

```bash
julia --project -e 'using Pkg; Pkg.test(; test_args=["--skip", "mcmc/gibbs.jl", "ad.jl"])'
```

CI matrix: Julia stable + min, Ubuntu/Windows/macOS, 1 and 2 threads.

`test/test_utils/sampler.jl` provides generic test helpers (`test_rng_respected`, `test_sampler_analytical`, `test_chain_logp_metadata`) that should work for any sampler. Beyond these, sampler-specific tests are needed to capture the properties you care about — there is no standardised test template yet.

## Architecture

### What lives here vs elsewhere

Most complexity is in DynamicPPL. Turing.jl contains:

  - **Sampler implementations** (`src/mcmc/`): HMC/NUTS/HMCDA (wrapping AdvancedHMC), MH (wrapping AdvancedMH), particle samplers SMC/PG/CSMC (implemented natively in `particle_mcmc.jl` on top of Libtask coroutines), ESS (wrapping EllipticalSliceSampling), SGLD/SGHMC, Emcee, and Gibbs.
  - **External sampler interface** (`src/mcmc/external_sampler.jl`): The `externalsampler()` wrapper lets any `AbstractMCMC.AbstractSampler` that implements `step` for `LogDensityModel` work with Turing models. This is the easier path for new samplers — it only requires a dependency on AbstractMCMC and the LogDensityProblems.jl interface, with no Turing internals. The tradeoff is less power: you can only interact with the model as a black-box log-density function, just like using `LogDensityFunction` directly.
  - **Variational inference** (`src/variational/`): Wraps AdvancedVI algorithms.
  - **Mode estimation** (`src/optimisation/`): MAP and MLE via Optimization.jl.
  - **Custom distributions** (`src/stdlib/`): `Flat`, `FlatPos`, `BinomialLogit`, `OrderedLogistic`, `LogPoisson`, and Dirichlet/Chinese Restaurant processes.

For how the model and inference machinery works under the hood, see the [DynamicPPL docs](https://turinglang.org/DynamicPPL.jl/stable/) and the [developer guides](https://turinglang.org/docs/developers/).

### Gibbs sampler

The Gibbs sampler (`src/mcmc/gibbs.jl`) is the most complex piece in Turing.jl. It threads a `VarNamedTuple` of raw values for all variables through the sweep. To step a component, it `condition`s the model on the values of every variable that component does not sample (`conditioned_values` picks them out), runs the component sampler, and merges the values it returns into the next component's conditioning set. The threaded `VarNamedTuple` is never mutated in place.

To plug a sampler into Gibbs, implement:

  - `gibbs_get_parameter_values(state)` — return a `VarNamedTuple` of the values of the variables this sampler is responsible for, leaving out `:=` quantities. The old name `gibbs_get_raw_values` still works, with a deprecation warning.
  - `gibbs_update_state!!(sampler, state, model, global_vals)` — update the sampler's state to reflect new conditioned values. For samplers that use `LogDensityFunction`, the helper `gibbs_recompute_ldf_and_params` handles the common case.
  - Optionally, `supports_gibbs(sampler)` — return `false` to disallow use in Gibbs (the default is `true`). The old name `isgibbscomponent` still works, with a deprecation warning.
  - Optionally, `allow_varying_dimension(sampler)` — return `true` if the sampler's own proposal can move between supports *within* a step (the default is `false`). See its docstring for what declaring it obliges the sampler to handle.
  - Optionally, `gibbs_update_state!!(sampler, state, model, global_vals, ::ReshapedBlock)` — the five-argument form, called instead of the four-argument one when another component's step has changed the block's parameter layout since this sampler last stepped, either by changing which variables it holds or by moving one to a distribution that links to a different width. The gate compares the linked width, measured by linking the value, not the distribution's family or its bijector's type: those are proxies that diverge from the layout, and keying on the family refused an adapting `NUTS` for a `Normal()`/`TDist(3)` branch whose block had not moved. A `truncated(Normal(); lower=a)` whose bound moves every sweep is likewise not a reshape. It defaults to throwing, so implementing it is how a sampler declares that it copes; there is no separate trait that could fall out of step with the implementation, and a sampler written before it existed keeps the safe answer. `MH`, `PG`/`CSMC` and `GibbsConditional` delegate to the four-argument form; a `Hamiltonian` that is not adapting rebuilds its parameter layout and phasepoint, which covers `HMC` and also `NUTS(0, δ)` and `HMCDA(0, δ, λ)`, whose states carry `NoAdaptation`. An adapting `NUTS` or `HMCDA`, `ESS` and `externalsampler` do not implement it, each carrying something sized for the block.
  - Optionally, `gibbs_get_stats(state)` — return a `NamedTuple` of the component's statistics for the chain (the default is empty). Gibbs drops component transitions, so statistics have to come off the state.

### Extension

`ext/TuringDynamicHMCExt` provides the DynamicHMC.jl integration (loaded when DynamicHMC is imported).

## Review Guidelines

### Use `OnlyAccsVarInfo`, not `VarInfo`

Sampler state should use `OnlyAccsVarInfo` (with appropriate accumulators), not `VarInfo`. `VarInfo` is being phased out across the ecosystem.

Most gradient-based samplers (HMC, NUTS, external samplers) go through `LogDensityFunction`, which handles the model interaction. `LogDensityFunction` works well when the model structure is static (the set of variables is fixed across evaluations) and the sampler only needs a scalar log-density value. However, LDF is hard to use when the sampler needs extra accumulators beyond log-probability — for example, MH uses custom accumulators to capture proposal distributions and linked values, so it works directly with `OnlyAccsVarInfo` + `init!!` instead. Either approach is fine; the key constraint is no `VarInfo`.

Note: "linked" and "unconstrained" are synonymous in this codebase. Linking transforms constrained parameters to unconstrained (Euclidean) space for gradient-based sampling.

### `VarNamedTuple` for parameter collections

Interfaces that accept or return named parameter collections should use `VarNamedTuple`, not `NamedTuple` or `Dict{VarName}`. `NamedTuple` and `Dict{VarName}` are accepted as user-facing input but should be converted to `VarNamedTuple` at the boundary (see `_to_varnamedtuple` in `src/common.jl`). Don't propagate them through internal code.

### `getlogjoint_internal` vs `getlogjoint`

Samplers operating in unconstrained space should use `getlogjoint_internal`, which includes the Jacobian correction from the linking transform. This is the default and what you almost always want. The exceptions are ESS (which needs the likelihood in constrained space, per the algorithm) and optimisation (where the Jacobian term should not influence the objective).

### AD backend handling

Gradient-based samplers accept an `adtype::ADTypes.AbstractADType` keyword (default: `AutoForwardDiff()`). When reviewing sampler code, check that `adtype` is threaded through to `LogDensityFunction` and not hardcoded. The AD backend is the user's choice, not the sampler's.

### `initial_params` conversion

User-facing functions accept `initial_params` as a convenience. `_convert_initial_params` in `src/common.jl` converts `NamedTuple`/`Dict{VarName}` to `InitFromParams`. Raw vectors are no longer supported and will error. Don't bypass this conversion or accept raw vectors in new code.

### Discrete variables

`Turing.Inference.allow_discrete_variables(sampler)` defaults to `true`. Gradient-based samplers (all `Hamiltonian` subtypes) override this to `false`. `_check_model` uses this to validate the model before sampling. If adding a new sampler that requires continuous variables, override `allow_discrete_variables` to return `false`.

### One component may not change another's dimension or existence

A component may never change the dimension of a variable belonging to another component, or
whether that variable exists at all, unless the two share it. Sharing is the only remedy: put
the deciding variable and what it decides in one block, or write a component that samples both.

A component moving another block's *support or distributional form* is a different matter and is
PERMITTED. That is a deliberate relaxation: each component's kernel stays invariant for its full
conditional, and what a support change can cost is irreducibility, which depends on whether
every reachable pair of supports overlaps — a question no single model evaluation decides. It is
therefore the caller's to establish. The `Gibbs` docstring carries the warning and a worked bad
example (`Uniform(0, 1)` against `Uniform(2, 3)`, absorbed, measured P(b=1) of 0.0/0.0/1.0
against 0.515 in one block); do not weaken that warning, and do not add an inferred refusal in
its place. Note the safe case is not distinguished by dimension or by family: the fatal example
holds both constant.

What is refused — a variable appearing or leaving — is enforced in `check_variable_set` on a
best-effort basis only. It compares the snapshots either side of a component's step, so a
crossing that was proposed and rejected leaves no trace and passes. In practice it is reliable —
on the dimension example in the `Gibbs` docstring it refused the split partition on all 40 seeds
tried, at 50, 300 and 2000 draws alike — but that is a fact about those chains, not a guarantee:
a decider that stays put for a whole run leaves the partition unexamined. Do not treat a
completed run as evidence that a partition is valid, and do not describe the check as a
guarantee in docs or error text. Ownership is tested by `_require_owned`:

  - whether a tilde statement executes at all — the variable appears or leaves.

The linked width is still recorded per tilde statement, because `block_fingerprint` needs it: a
distribution change can move a block's *linked* dimension even when its values keep their shape,
and a component that cannot rebuild for that is refused by `gibbs_update_state!!` for a
`ReshapedBlock`. That refusal is about layout, not about the correctness of the chain.

### Conditioned variables are observations

Gibbs conditions a component on every variable it does not sample, so those variables reach `tilde_observe!!` and particle samplers (PG/CSMC) reweight on them. That is what makes the component's target correct: a conditioned variable the target depends on must reweight the sweep, while one it does not contributes the same increment to every particle, which ESS-gated resampling ignores. Do not try to route them through `tilde_assume!!` to avoid resampling.

## Contributing

  - Non-breaking changes target `main`; breaking changes target the `breaking` branch.
  - Julia ≥ 1.10.8 required (see `[compat]` in `Project.toml`).
  - `HISTORY.md`: one line for a bugfix or internal change. Only a breaking change or new feature earns more, and only what a user needs to act on it: what broke, and the old → new form. The mechanism and any measurements belong in the commit and the PR.
