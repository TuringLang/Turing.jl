<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://turinglang.org/assets/logo/turing-logo-dark.svg">
    <img src="https://turinglang.org/assets/logo/turing-logo-light.svg" alt="Turing.jl logo" width="300">
  </picture>
</p>
<p align="center"><i>Bayesian inference with probabilistic programming</i></p>
<p align="center">
<a href="https://turinglang.org/"><img src="https://img.shields.io/badge/docs-tutorials-blue.svg" alt="Tutorials" /></a>
<a href="https://turinglang.org/Turing.jl/stable"><img src="https://img.shields.io/badge/docs-API-blue.svg" alt="API docs" /></a>
<a href="https://github.com/TuringLang/Turing.jl/actions/workflows/Tests.yml"><img src="https://github.com/TuringLang/Turing.jl/actions/workflows/Tests.yml/badge.svg" alt="Tests" /></a>
<a href="https://codecov.io/gh/TuringLang/Turing.jl"><img src="https://codecov.io/gh/TuringLang/Turing.jl/branch/main/graph/badge.svg" alt="Code Coverage" /></a>
<a href="https://github.com/SciML/ColPrac"><img src="https://img.shields.io/badge/ColPrac-Contributor%27s%20Guide-blueviolet" alt="ColPrac: Contributor's Guide on Collaborative Practices for Community Packages" /></a>
</p>

`Turing.jl` is a Julia probabilistic programming package for Bayesian and
likelihood-based inference. A Turing model specifies a joint probability distribution
as executable Julia code. The same model can then be used for Markov chain Monte Carlo,
variational inference, maximum likelihood estimation, or maximum a posteriori
estimation.

Turing is the user-facing entry point to the TuringLang ecosystem.
[DynamicPPL.jl](https://github.com/TuringLang/DynamicPPL.jl) represents and evaluates
models, while Turing connects them to inference algorithms. Markov chain Monte Carlo
uses [AbstractMCMC.jl](https://github.com/TuringLang/AbstractMCMC.jl) and samplers such as
[AdvancedMH.jl](https://github.com/TuringLang/AdvancedMH.jl) and
[AdvancedHMC.jl](https://github.com/TuringLang/AdvancedHMC.jl). Variational inference
uses [AdvancedVI.jl](https://github.com/TuringLang/AdvancedVI.jl), and mode estimation
uses [Optimization.jl](https://github.com/SciML/Optimization.jl).

Gradient-based algorithms require derivatives of the model's log density. Turing's
preferred automatic differentiation backends are `ForwardDiff.jl` and `Mooncake.jl`,
which are integrated through their public APIs. Further backends are available through
`DifferentiationInterface.jl`.

## Get started

Install Julia 1.10.8 or later from the [official Julia
website](https://julialang.org/install/). Then open a Julia REPL and run:

```julia
julia> using Pkg; Pkg.add("Turing")
```

The following example places priors on an intercept, slope, and residual scale, then
samples their posterior distribution:

```julia
julia> using Random, Turing

julia> rng = Xoshiro(1)

julia> @model function linear_regression(x)
           # Priors
           α ~ Normal(0, 1)
           β ~ Normal(0, 1)
           σ ~ truncated(Cauchy(0, 3); lower=0)

           # Likelihood
           μ = α .+ β .* x
           y ~ MvNormal(μ, σ^2 * I)
       end

julia> x = range(-1, 1; length=20)

julia> y = 1 .+ 2 .* x .+ 0.2 .* randn(rng, length(x))

julia> posterior = linear_regression(x) | (; y = y)

julia> chain = sample(rng, posterior, NUTS(), 1000)
```

The example first generates synthetic observations with intercept 1, slope 2, and noise
standard deviation 0.2. The `| (; y = y)` expression then conditions the model on those
observations. `NUTS()` selects the No-U-Turn sampler, and `chain` contains 1,000
posterior draws of `α`, `β`, and `σ`.

## Documentation and discussion

The [TuringLang documentation](https://turinglang.org/) provides tutorials, while the
[Turing.jl documentation](https://turinglang.org/Turing.jl/stable) is the API reference.
The [TuringLang newsletter](https://turinglang.org/news/) reports work across the
ecosystem. Changes to Turing.jl are recorded in
[`HISTORY.md`](https://github.com/TuringLang/Turing.jl/blob/main/HISTORY.md) and the
[GitHub releases](https://github.com/TuringLang/Turing.jl/releases).

Technical discussion takes place in the [`#turing` channel of Julia
Slack](https://julialang.slack.com/archives/CCYDC34A0) and under the [`turing` tag on Julia
Discourse](https://discourse.julialang.org/tag/turing). The Julia website provides
[Slack invitations](https://julialang.org/slack/).

## Project scope

Turing is maintained as grant-funded research software. It prioritises correctness and
stability over broad feature coverage.

Reproducible cases of incorrect results or unexpected failures within the documented
scope guide further work. Capacity for maintenance and review is necessarily limited.

## Contributing

Discuss proposed features in an [issue](https://github.com/TuringLang/Turing.jl/issues)
before implementation so that their fit can be assessed. Focused bug fixes and small
changes may be submitted directly as pull requests. Bug reports need not identify the
correct TuringLang repository in advance; maintainers can transfer issues when
necessary.

Pull requests for non-breaking changes target `main`; breaking changes target
`breaking`. Reviewer privileges are reserved for sustained, substantive contributors
and people invited by a team member. If an issue or pull request has received no
response, ping `@TuringLang/maintainers`.

## Citing Turing.jl

If you use Turing.jl in published work, please cite:

[**Turing.jl: A General-Purpose Probabilistic Programming Language**](https://doi.org/10.1145/3711897)<br>
Tor Erlend Fjelde, Kai Xu, David Widmann, Mohamed Tarek, Cameron Pfiffer, Martin Trapp, Seth D. Axen, Xianda Sun, Markus Hauru, Penelope Yong, Will Tebbutt, Zoubin Ghahramani, Hong Ge<br>
ACM Transactions on Probabilistic Machine Learning, 1(3):1–48, 2025.

[**Turing: A Language for Flexible Probabilistic Inference**](https://proceedings.mlr.press/v84/ge18b.html)<br>
Hong Ge, Kai Xu, Zoubin Ghahramani<br>
Proceedings of the Twenty-First International Conference on Artificial Intelligence and Statistics, PMLR 84:1682–1690, 2018.

<details>

<summary>Expand for BibTeX</summary>

```bibtex
@article{10.1145/3711897,
  author = {Fjelde, Tor Erlend and Xu, Kai and Widmann, David and Tarek, Mohamed and Pfiffer, Cameron and Trapp, Martin and Axen, Seth D. and Sun, Xianda and Hauru, Markus and Yong, Penelope and Tebbutt, Will and Ghahramani, Zoubin and Ge, Hong},
  title = {Turing.jl: A General-Purpose Probabilistic Programming Language},
  journal = {ACM Trans. Probab. Mach. Learn.},
  year = {2025},
  volume = {1},
  number = {3},
  pages = {1--48},
  month = aug,
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  doi = {10.1145/3711897},
  url = {https://doi.org/10.1145/3711897},
}

@inproceedings{pmlr-v84-ge18b,
  author = {Ge, Hong and Xu, Kai and Ghahramani, Zoubin},
  title = {Turing: A Language for Flexible Probabilistic Inference},
  booktitle = {Proceedings of the Twenty-First International Conference on Artificial Intelligence and Statistics},
  editor = {Storkey, Amos and Perez-Cruz, Fernando},
  series = {Proceedings of Machine Learning Research},
  volume = {84},
  pages = {1682--1690},
  year = {2018},
  month = {09--11 Apr},
  publisher = {PMLR},
  pdf = {http://proceedings.mlr.press/v84/ge18b/ge18b.pdf},
  url = {https://proceedings.mlr.press/v84/ge18b.html},
}
```

</details>
