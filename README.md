<p align="center"><img src="https://turinglang.org/assets/logo/turing-logo-light.svg" alt="Turing.jl logo" width="300" /></p>
<p align="center"><i>Bayesian inference with probabilistic programming</i></p>
<p align="center">
<a href="https://turinglang.org/"><img src="https://img.shields.io/badge/docs-tutorials-blue.svg" alt="Tutorials" /></a>
<a href="https://turinglang.org/Turing.jl/stable"><img src="https://img.shields.io/badge/docs-API-blue.svg" alt="API docs" /></a>
<a href="https://github.com/TuringLang/Turing.jl/actions/workflows/Tests.yml"><img src="https://github.com/TuringLang/Turing.jl/actions/workflows/Tests.yml/badge.svg" alt="Tests" /></a>
<a href="https://codecov.io/gh/TuringLang/Turing.jl"><img src="https://codecov.io/gh/TuringLang/Turing.jl/branch/main/graph/badge.svg" alt="Code Coverage" /></a>
<a href="https://github.com/SciML/ColPrac"><img src="https://img.shields.io/badge/ColPrac-Contributor%27s%20Guide-blueviolet" alt="ColPrac: Contributor's Guide on Collaborative Practices for Community Packages" /></a>
</p>

## Get started

Install Julia (see [the official Julia website](https://julialang.org/install/); you will need at least Julia 1.10 for the latest version of Turing.jl).
Then, launch a Julia REPL and run:

```julia
julia> using Pkg; Pkg.add("Turing")
```

You can define models using the `@model` macro, and then perform Markov chain Monte Carlo sampling using the `sample` function:

```julia
julia> using Turing

julia> @model function my_first_model(data)
           mean ~ Normal(0, 1)
           sd ~ truncated(Cauchy(0, 3); lower=0)
           data ~ Normal(mean, sd)
       end

julia> model = my_first_model(randn())

julia> chain = sample(model, NUTS(), 1000)
```

You can find the main TuringLang documentation at [**https://turinglang.org**](https://turinglang.org), which contains general information about Turing.jl's features, as well as a variety of tutorials with examples of Turing.jl models.

API documentation for Turing.jl is specifically available at [**https://turinglang.org/Turing.jl/stable**](https://turinglang.org/Turing.jl/stable/).

## Contributing

### Issues

If you find any bugs or unintuitive behaviour when using Turing.jl, please do [open an issue](https://github.com/TuringLang/Turing.jl/issues)!
Please don't worry about finding the correct repository for the issue; we can migrate the issue to the appropriate repository if we need to.

### Pull requests

We are of course also very happy to receive pull requests.
If you are unsure about whether a particular feature would be welcome, you can open an issue for discussion first.

When opening a PR, non-breaking releases (patch versions) should target the `main` branch.
Breaking releases (minor version) should target the `breaking` branch.

If you have not received any feedback on an issue or PR for a while, please feel free to ping `@TuringLang/maintainers` in a comment.

## Other channels

The Turing.jl userbase tends to be most active on the [`#turing` channel of Julia Slack](https://julialang.slack.com/archives/CCYDC34A0).
If you do not have an invitation to Julia's Slack, you can get one from [the official Julia website](https://julialang.org/slack/).

There are also often threads on [Julia Discourse](https://discourse.julialang.org) (you can search using, e.g., [the `turing` tag](https://discourse.julialang.org/tag/turing)).

## What's changed recently?

We publish a fortnightly newsletter summarising recent updates in the TuringLang ecosystem, which you can view on [our website](https://turinglang.org/news/), [GitHub](https://github.com/TuringLang/Turing.jl/issues/2498), or [Julia Slack](https://julialang.slack.com/archives/CCYDC34A0).

For Turing.jl specifically, you can see a full changelog in [`HISTORY.md`](https://github.com/TuringLang/Turing.jl/blob/main/HISTORY.md) or [our GitHub releases](https://github.com/TuringLang/Turing.jl/releases).

## Where does Turing.jl sit in the TuringLang ecosystem?

Turing.jl is the main entry point for users, and seeks to provide a unified, convenient interface to all of the functionality in the TuringLang (and broader Julia) ecosystem.

In particular, it takes the ability to specify probabilistic models with [DynamicPPL.jl](https://github.com/TuringLang/DynamicPPL.jl), and combines it with a number of inference algorithms, such as:

  - Markov Chain Monte Carlo (both an abstract interface: [AbstractMCMC.jl](https://github.com/TuringLang/AbstractMCMC.jl), and individual samplers, such as [AdvancedMH.jl](https://github.com/TuringLang/AdvancedMH.jl), [AdvancedHMC.jl](https://github.com/TuringLang/AdvancedHMC.jl), and more).
  - Variational inference using [AdvancedVI.jl](https://github.com/TuringLang/AdvancedVI.jl).
  - Maximum likelihood and maximum a posteriori estimation, which rely on SciML's [Optimization.jl interface](https://github.com/SciML/Optimization.jl).

## Citing Turing.jl

If you have used Turing.jl in your work, we would be very grateful if you could cite the following:

[**Turing.jl: a general-purpose probabilistic programming language**](https://doi.org/10.1145/3711897)  
Tor Erlend Fjelde, Kai Xu, David Widmann, Mohamed Tarek, Cameron Pfiffer, Martin Trapp, Seth D. Axen, Xianda Sun, Markus Hauru, Penelope Yong, Will Tebbutt, Zoubin Ghahramani, Hong Ge  
ACM Transactions on Probabilistic Machine Learning, 2025 (_Just Accepted_)  

[**Turing: A Language for Flexible Probabilistic Inference**](https://proceedings.mlr.press/v84/ge18b.html)  
Hong Ge, Kai Xu, Zoubin Ghahramani  
Proceedings of the Twenty-First International Conference on Artificial Intelligence and Statistics, PMLR 84:1682-1690, 2018.

<details>

<summary>Expand for BibTeX</summary>

```bibtex
@article{10.1145/3711897,
author = {Fjelde, Tor Erlend and Xu, Kai and Widmann, David and Tarek, Mohamed and Pfiffer, Cameron and Trapp, Martin and Axen, Seth D. and Sun, Xianda and Hauru, Markus and Yong, Penelope and Tebbutt, Will and Ghahramani, Zoubin and Ge, Hong},
title = {Turing.jl: a general-purpose probabilistic programming language},
year = {2025},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3711897},
doi = {10.1145/3711897},
note = {Just Accepted},
journal = {ACM Trans. Probab. Mach. Learn.},
month = feb,
}

@InProceedings{pmlr-v84-ge18b,
  title = 	 {Turing: A Language for Flexible Probabilistic Inference},
  author = 	 {Ge, Hong and Xu, Kai and Ghahramani, Zoubin},
  booktitle = 	 {Proceedings of the Twenty-First International Conference on Artificial Intelligence and Statistics},
  pages = 	 {1682--1690},
  year = 	 {2018},
  editor = 	 {Storkey, Amos and Perez-Cruz, Fernando},
  volume = 	 {84},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {09--11 Apr},
  publisher =    {PMLR},
  pdf = 	 {http://proceedings.mlr.press/v84/ge18b/ge18b.pdf},
  url = 	 {https://proceedings.mlr.press/v84/ge18b.html},
}
```

</details>
