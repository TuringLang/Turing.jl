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

Gibbs owns the assignment of variables to component samplers. Repeating that assignment
inside a component risks updating or scoring the wrong variables; see
`VarNamedTuple` for parameter collections below.
`Gibbs(@varname(x) => MH(cov_matrix))` applies `cov_matrix` to the complete linked vector of
the conditioned `x` block.

To plug a sampler into Gibbs, implement:

  - `gibbs_get_parameter_values(state)` — return a `VarNamedTuple` of the values of the variables this sampler is responsible for, leaving out `:=` quantities. The old name `gibbs_get_raw_values` still works, with a deprecation warning.
  - `gibbs_update_state!!(sampler, state, model, global_vals)` — update the sampler's state to reflect new conditioned values. For samplers that use `LogDensityFunction`, the helper `gibbs_recompute_ldf_and_params` handles the common case.
  - Optionally, `supports_gibbs(sampler)` — return `false` to disallow use in Gibbs (the default is `true`). The old name `isgibbscomponent` still works, with a deprecation warning.
  - Optionally, `allow_varying_dimension(sampler)` — return `true` if the sampler's own proposal can move between supports *within* a step (the default is `false`). See its docstring for what declaring it obliges the sampler to handle.
  - Optionally, `gibbs_update_state!!(sampler, state, model, global_vals, ::ReshapedBlock)` — the five-argument form, called instead of the four-argument one when another component has changed the block's parameter layout since this sampler last stepped. It defaults to throwing, so implementing it *is* the declaration: no trait can fall out of step with it, and a sampler predating it keeps the safe answer.
  - Optionally, `keeps_linked_layout(sampler)` — return `false` if the sampler holds no parameter vector in linked space (the default is `true`). Its block is then compared at the values' native shape, deriving no `Bijectors` transform, which a distribution defining no link cannot supply anyway. `MH`, `PG`/`CSMC` and `GibbsConditional` answer `false`.
  - Optionally, `gibbs_get_stats(state)` — return a `NamedTuple` of the component's statistics for the chain (the default is empty). Gibbs drops component transitions, so statistics have to come off the state.

The gate compares each tilde's linked width, measured by linking the value, not the family or
the bijector's type: those are proxies that diverge from the layout, and keying on the family
refused an adapting `NUTS` for a `Normal()`/`TDist(3)` branch whose block had not moved. An
unchanged distribution short-circuits, and a component answering `false` to `keeps_linked_layout`
never measures at all, so nothing is derived unless it has to be. A
`truncated(Normal(); lower=a)` with a moving bound is not a reshape either. `MH`, `PG`/`CSMC`
and `GibbsConditional` delegate to the four-argument form, and a non-adapting `Hamiltonian`
rebuilds (`HMC`, `NUTS(0, δ)`, `HMCDA(0, δ, λ)`). An adapting `NUTS` or `HMCDA`, `ESS` and
`externalsampler` do not, each carrying something sized for the block.

### Extension

`ext/TuringDynamicHMCExt` provides the DynamicHMC.jl integration (loaded when DynamicHMC is imported).

## Review Guidelines

### Use `OnlyAccsVarInfo`, not `VarInfo`

Sampler state should use `OnlyAccsVarInfo` (with appropriate accumulators), not `VarInfo`. `VarInfo` is being phased out across the ecosystem.

Most gradient-based samplers (HMC, NUTS, external samplers) go through `LogDensityFunction`, which handles the model interaction. `LogDensityFunction` works well when the model structure is static (the set of variables is fixed across evaluations) and the sampler only needs a scalar log-density value. However, LDF is hard to use when the sampler needs extra accumulators beyond log-probability. `MH()` works directly with `OnlyAccsVarInfo` + `init!!` because it draws proposals from the model prior; `MH(cov_matrix)` delegates to an external sampler and uses `LogDensityFunction`. Either approach is fine; the key constraint is no `VarInfo`.

Say "linked" and "unlinked", not "unconstrained" and "constrained": linking is what the API is named after (`to_linked_vec`, `UnlinkAll`, `LinkAll`), so the two vocabularies cannot both track it. Linking transforms parameters to unconstrained (Euclidean) space for gradient-based sampling.

Three exceptions, all deliberate. `unconstrained` is the public keyword of `externalsampler` and `vi`, and `AbstractMCMC.requires_unconstrained_space` is upstream, so prose about those keeps their word. A docstring may define "linked" in terms of "unconstrained" once, for a reader who knows only the latter. And in `src/optimisation/` "constraints" already means the user's `lb`/`ub` bounds -- an "unconstrained mode" there is one no bound reached, nothing to do with linking -- so never reach for that word to mean linked space.

### `VarNamedTuple` for parameter collections

Interfaces that accept or return named parameter collections should use `VarNamedTuple`, not `NamedTuple` or `Dict{VarName}`. `NamedTuple` and `Dict{VarName}` are accepted as user-facing input but should be converted to `VarNamedTuple` at the boundary (see `_to_varnamedtuple` in `src/common.jl`). Don't propagate them through internal code.

Do not infer executed `~` sites from `haskey` or `keys` on a parameter
`VarNamedTuple`: `haskey(values, @varname(x))` can match stored descendants, and one ranged
site can be stored as several keys. Treating these keys as sampling sites can score a
proposal against the wrong values, giving incorrect acceptance probabilities and
potentially biased inference.

### `getlogjoint_internal` vs `getlogjoint`

Samplers operating in unconstrained space should use `getlogjoint_internal`, which includes the Jacobian correction from the linking transform. This is the default and what you almost always want. The exceptions are ESS (which needs the likelihood in constrained space, per the algorithm) and optimisation (where the Jacobian term should not influence the objective).

### AD backend handling

Gradient-based samplers accept an `adtype::ADTypes.AbstractADType` keyword (default: `AutoForwardDiff()`). When reviewing sampler code, check that `adtype` is threaded through to `LogDensityFunction` and not hardcoded. The AD backend is the user's choice, not the sampler's.

### `initial_params` conversion

User-facing functions accept `initial_params` as a convenience. `_convert_initial_params` in `src/common.jl` converts `NamedTuple`/`Dict{VarName}` to `InitFromParams`. Raw vectors are no longer supported and will error. Don't bypass this conversion or accept raw vectors in new code.

### Discrete variables

`Turing.Inference.allow_discrete_variables(sampler)` defaults to `true`. Gradient-based samplers (all `Hamiltonian` subtypes) override this to `false`. `_check_model` uses this to validate the model before sampling. If adding a new sampler that requires continuous variables, override `allow_discrete_variables` to return `false`.

### One component may not change another's dimension or existence

A component may never change the dimension of a variable belonging to another, or whether it
exists, unless the two share it. The only remedy is sharing: put the deciding variable and what
it decides in one block, or write a component that samples both.

Moving another block's *support or distributional form* is PERMITTED, deliberately. Each kernel
stays invariant for its full conditional; what breaks is irreducibility, which no model
evaluation decides, so it is the caller's to establish. The `Gibbs` docstring carries the
warning and a worked bad example — do not weaken it, and do not add an inferred refusal in its
place. Neither dimension nor family separates the safe case from the fatal one; that example
holds both constant.

The refusal is best-effort: `check_variable_set` compares snapshots either side of a step, so a
rejected crossing leaves no trace and a decider that never moves leaves the partition
unexamined. Never call it a guarantee in docs or error text, and never read a completed run as
evidence that a partition is valid. `_require_owned` tests one thing: whether a tilde statement
executes at all.

`block_fingerprint` records each tilde's linked width because a distribution change can move a
block's linked dimension while its values keep their shape; a component that cannot rebuild is
refused with a `ReshapedBlock`. Layout, not chain correctness.

### Gibbs refuses a `missing` model argument

Conditioning cannot reach a variable that is a model argument: the compiler reads the argument
directly for such a tilde, so the `missing` arrives at the likelihood and throws. This is a
capability `GibbsContext` had and conditioning does not, so Gibbs refuses instead —
`check_no_missing_arguments`, over `model.args` AND `model.defaults`, whole arguments and array
elements alike. Do not narrow it on the strength of a `GibbsConditional` test: that sampler
evaluates without a log-likelihood accumulator and masks the failure. Restoring the capability
needs DynamicPPL.jl#1462 (unmerged), and `missing` as a latent marker is due for deprecation
anyway (DynamicPPL.jl#1464). Keep the check in Gibbs and out of `_check_model`: other samplers
take these models.

### Conditioned variables are observations

Gibbs conditions a component on every variable it does not sample, so those variables reach `tilde_observe!!` and particle samplers (PG/CSMC) reweight on them. That is what makes the component's target correct: a conditioned variable the target depends on must reweight the sweep, while one it does not contributes the same increment to every particle, which ESS-gated resampling ignores. Do not try to route them through `tilde_assume!!` to avoid resampling.

## Contributing

  - Non-breaking changes target `main`; breaking changes target the `breaking` branch.
  - Julia ≥ 1.10.8 required (see `[compat]` in `Project.toml`).
  - `HISTORY.md`: one line for a bugfix or internal change. Only a breaking change or new feature earns more, and only what a user needs to act on it: what broke, and the old → new form. The mechanism and any measurements belong in the commit and the PR. End each entry with its PR: `([#1234](https://github.com/TuringLang/Turing.jl/pull/1234))`.
