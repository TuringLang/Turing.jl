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

`Turing.jl` is a general-purpose probabilistic programming package for Bayesian and
likelihood-based inference. Implemented in Julia, it is designed to interoperate with the
language's scientific computing ecosystem. Models can be written with
[`@model`](https://github.com/TuringLang/DynamicPPL.jl), in BUGS syntax via
[JuliaBUGS.jl](https://github.com/TuringLang/JuliaBUGS.jl), or graphically with
[DoodlePPL](https://turinglang.org/JuliaBUGS.jl/DoodlePPL/), and support MCMC, variational
inference, maximum likelihood and MAP estimation, and customised inference through Turing's
log-density and gradient interface.

Turing's preferred automatic differentiation backends for gradient-based algorithms
are [ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl) and
[Mooncake.jl](https://github.com/chalk-lab/Mooncake.jl). Other backends, such as
[Enzyme.jl](https://github.com/EnzymeAD/Enzyme.jl), are available through
[DifferentiationInterface.jl](https://github.com/JuliaDiff/DifferentiationInterface.jl).

## Get started

Install Julia 1.10.8 or later from the [official Julia
website](https://julialang.org/install/). Then open a Julia REPL and run:

```julia
julia> using Pkg; Pkg.add("Turing")
```

The following example places priors on an intercept, slope, and residual scale, then
samples the posterior distribution of these parameters:

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

The example generates synthetic observations, conditions the model on them with
`| (; y = y)`, and draws 1,000 posterior samples of `α`, `β`, and `σ` using the
No-U-Turn sampler.

## Documentation and discussion

See the [TuringLang tutorials](https://turinglang.org/),
[Turing.jl API reference](https://turinglang.org/Turing.jl/stable), and
[TuringLang newsletter](https://turinglang.org/news/). Changes are recorded in
[`HISTORY.md`](https://github.com/TuringLang/Turing.jl/blob/main/HISTORY.md); published
releases are available on the [GitHub releases
page](https://github.com/TuringLang/Turing.jl/releases).

For technical discussion, use the [`#turing` channel on Julia
Slack](https://julialang.slack.com/archives/CCYDC34A0) ([request an
invitation](https://julialang.org/slack/)) or the [`turing` tag on Julia
Discourse](https://discourse.julialang.org/tag/turing).

## Project scope

Turing is grant-funded research software that prioritises correctness and stability
over broad feature coverage. New features are often developed through research projects
or collaborations. Reproducible reports of incorrect results or unexpected failures in
documented functionality guide further work.

## Contributing

Discuss proposed features in an [issue](https://github.com/TuringLang/Turing.jl/issues)
before implementation. Focused bug fixes and small changes may be submitted directly
as pull requests. Maintainers can transfer bug reports filed in the wrong TuringLang
repository.

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
