# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

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

## Architecture

### What lives here vs elsewhere

Most complexity is in DynamicPPL. Turing.jl contains:

  - **Sampler implementations** (`src/mcmc/`): HMC/NUTS/HMCDA (wrapping AdvancedHMC), MH (wrapping AdvancedMH), particle samplers SMC/PG/CSMC (wrapping AdvancedPS), ESS (wrapping EllipticalSliceSampling), SGLD/SGHMC, Emcee, and Gibbs.
  - **External sampler interface** (`src/mcmc/external_sampler.jl`): The `externalsampler()` wrapper lets any `AbstractMCMC.AbstractSampler` that implements `step` for `LogDensityModel` work with Turing models.
  - **Variational inference** (`src/variational/`): Wraps AdvancedVI algorithms.
  - **Mode estimation** (`src/optimisation/`): MAP and MLE via Optimization.jl.
  - **Custom distributions** (`src/stdlib/`): `Flat`, `FlatPos`, `BinomialLogit`, `OrderedLogistic`, `LogPoisson`, and Dirichlet/Chinese Restaurant processes.

For how the model and inference machinery works under the hood, see the [DynamicPPL docs](https://turinglang.org/DynamicPPL.jl/stable/) and the [developer guides](https://turinglang.org/docs/developers/).

### Gibbs sampler

The Gibbs sampler (`src/mcmc/gibbs.jl`) is the most complex piece in Turing.jl. It maintains a global `VarNamedTuple` of raw values for all variables. On each iteration, it conditions the model on the non-target variables via `GibbsContext`, runs the component sampler, and updates the global state.

To plug a sampler into Gibbs, implement:

  - `gibbs_get_raw_values(state)` — return a `VarNamedTuple` of raw values for the variables this sampler is responsible for.
  - `gibbs_update_state!!(sampler, state, model, global_vals)` — update the sampler's state to reflect new conditioned values. For samplers that use `LogDensityFunction`, the helper `gibbs_recompute_ldf_and_params` handles the common case.
  - Optionally, `isgibbscomponent(sampler)` — return `false` to disallow use in Gibbs (the default is `true`).

### Extension

`ext/TuringDynamicHMCExt` provides the DynamicHMC.jl integration (loaded when DynamicHMC is imported).

## Review Guidelines

### Use `OnlyAccsVarInfo`, not `VarInfo`

Sampler state should use `OnlyAccsVarInfo` (with appropriate accumulators), not `VarInfo`. `VarInfo` is being phased out across the ecosystem. Most gradient-based samplers (HMC, NUTS, external samplers) go through `LogDensityFunction`, which handles the model interaction. But not all samplers can use `LogDensityFunction` — MH, for example, works directly with `OnlyAccsVarInfo` + `init!!` because it needs per-variable proposal control. Either approach is fine; the key constraint is no `VarInfo`.

### `VarNamedTuple` for parameter collections

Interfaces that accept or return named parameter collections should use `VarNamedTuple`, not `NamedTuple` or `Dict{VarName}`. `NamedTuple` and `Dict{VarName}` are accepted as user-facing input but should be converted to `VarNamedTuple` at the boundary (see `_to_varnamedtuple` in `src/common.jl`). Don't propagate them through internal code.

### AD backend handling

All samplers accept an `adtype::ADTypes.AbstractADType` keyword (default: `AutoForwardDiff()`). When reviewing sampler code, check that `adtype` is threaded through to `LogDensityFunction` and not hardcoded. The AD backend is the user's choice, not the sampler's.

### `initial_params` conversion

User-facing functions accept `initial_params` as a convenience. `_convert_initial_params` in `src/common.jl` converts `NamedTuple`/`Dict{VarName}` to `InitFromParams`. Raw vectors are no longer supported and will error. Don't bypass this conversion or accept raw vectors in new code.

### Discrete variables

`allow_discrete_variables(sampler)` defaults to `true`. Gradient-based samplers (all `Hamiltonian` subtypes) override this to `false`. `_check_model` uses this to validate the model before sampling. If adding a new sampler that requires continuous variables, override `allow_discrete_variables` to return `false`.

### GibbsContext is not ConditionContext

`GibbsContext` is distinct from `condition`/`ConditionContext`. For non-target variables, `GibbsContext.tilde_assume!!` calls `tilde_observe!!` — this means particle samplers (PG/CSMC) will correctly resample on conditioned variables. The key difference from `condition` is that `GibbsContext` obtains the conditioned values from the global `VarNamedTuple` rather than from the model's conditioning, and it handles the bookkeeping needed for Gibbs (e.g. updating the global VNT when new variables appear).

## Contributing

  - Non-breaking changes target `main`; breaking changes target the `breaking` branch.
  - Julia ≥ 1.10.8 required (see `[compat]` in `Project.toml`).
