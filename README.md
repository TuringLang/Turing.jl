<p align="center"><img src="https://raw.githubusercontent.com/TuringLang/turinglang.github.io/refs/heads/main/assets/images/turing-logo.svg" alt="Turing.jl logo" width="200" /></p>
<h1 align="center">Turing.jl</h1>
<p align="center">
[![Tutorials](https://img.shields.io/badge/docs-tutorials-blue.svg)](https://turinglang.org/Turing.jl/stable)
[![API docs](https://img.shields.io/badge/docs-API-blue.svg)](https://turinglang.org/Turing.jl/stable)
[![Tests](https://github.com/TuringLang/Turing.jl/actions/workflows/Tests.yml/badge.svg)](https://github.com/TuringLang/Turing.jl/actions/workflows/Tests.yml)
[![Coverage Status](https://coveralls.io/repos/github/TuringLang/Turing.jl/badge.svg?branch=main)](https://coveralls.io/github/TuringLang/Turing.jl?branch=main)
[![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor%27s%20Guide-blueviolet)](https://github.com/SciML/ColPrac)
</p>


**Turing.jl** is a framework for probabilistic programming and Bayesian inference in Julia.

## 📚 Documentation

**https://turinglang.org** contains the documentation for the broader TuringLang ecosystem.

**https://turinglang.org/Turing.jl/** specifically contains the API documentation for anything exported by Turing.jl.

## 🛠️ Contributing

### Issues

If you find any bugs or unintuitive behaviour when using Turing.jl, please do [open an issue](https://github.com/TuringLang/Turing.jl/issues)!
Please don't worry about finding the correct repository for the issue; we can migrate the issue to the appropriate repository if we need to.

### Pull requests

We are of course also very happy to receive pull requests.
If you are unsure about whether a particular feature would be welcome, you can open an issue for discussion first.

When opening a PR, non-breaking releases (patch versions) should target the `main` branch.
Breaking releases (minor version) should target the `breaking` branch.

If you have not received any feedback on an issue or PR for a while, please feel free to ping `@TuringLang/maintainers` in a comment.

### Discussions

This repository also has a [Discussions page](https://github.com/TuringLang/Turing.jl/discussions), where you can create discussions and ask questions about statistical applications and theory.
In practice, we don't monitor this as often: you're more likely to get a response through an issue.

## 💬 Other channels

The Turing.jl userbase tends to be most active on the [`#turing` channel of Julia Slack](https://julialang.slack.com/archives/CCYDC34A0).
If you do not have an invitation to Julia's Slack, you can get one from [the official Julia website](https://julialang.org/slack/).

There are also often threads on [Julia Discourse](https://discourse.julialang.org) (you can search using, e.g., [the `turing` tag](https://discourse.julialang.org/tag/turing)).

We are most active on GitHub, but we do also keep an eye on both Slack and Discourse.

## 🧩 Where does Turing.jl sit in the TuringLang ecosystem?

Turing.jl is the main entry point for users, and seeks to provide a unified, convenient interface to all of the functionality in the TuringLang (and broader Julia) ecosystem.

In particular, it takes the ability to specify probabilistic models with [DynamicPPL.jl](https://github.com/TuringLang/DynamicPPL.jl), and combines it with a number of inference algorithms, such as:

  - Markov Chain Monte Carlo (both an abstract interface: [AbstractMCMC.jl](https://github.com/TuringLang/AbstractMCMC.jl), and individual samplers, such as [AdvancedMH.jl](https://github.com/TuringLang/AdvancedMH.jl), [AdvancedHMC.jl](https://github.com/TuringLang/AdvancedHMC.jl), and more).
  - Variational inference using [AdvancedVI.jl](https://github.com/TuringLang/AdvancedVI.jl).
  - Mode estimation techniques, which rely on SciML's [Optimization.jl interface](https://github.com/SciML/Optimization.jl).
