---
title: Turing.jl
layout: splash
permalink: /splash/

header:
  overlay_color: "#FFF"
  # overlay_filter: "0.0"
  # overlay_image: /assets/turing-logo-wide.svg
  actions:
    - label: "Get Started"
      url: "http://turing.ml/docs/get-started/"
    - label: "Documentation"
      url: "http://turing.ml/docs/"
    - label: "Tutorials"
      url: "http://turing.ml/tutorials/"
excerpt: "**Turing** is a *universal* probabilistic programming language with an intuitive modelling interface, composable probabilistic inference, and computational scalability."

intro:
  - excerpt: 'Turing provides Hamiltonian Monte Carlo and particle MCMC sampling algorithms for complex posterior distributions ideal for distributions involving discrete variables and stochastic control flows.'

current-features:
  - title: 'Current Features'

main-feature_row:
  - i_class: "fas fa-stroopwafel"
    alt: "placeholder image 1"
    title: "Placeholder 1"
    excerpt: "This is some sample content that goes here with **Markdown** formatting."
  - i_class: "fas fa-brain"
    alt: "placeholder image 2"
    title: "Placeholder 2"
    excerpt: "This is some sample content that goes here with **Markdown** formatting."
    url: "#test-link"
    btn_label: "Read More"
    btn_class: "btn--primary"
  - image_path: /assets/images/unsplash-gallery-image-3-th.jpg
    title: "Placeholder 3"
    excerpt: "This is some sample content that goes here with **Markdown** formatting."

feature_row:
  - title: "Universal"
    excerpt: "Turing is a universal probabilistic programming language with an intuitive modelling interface. Write models with ease in Julia's straightforward syntax."
  # - title: "Large Sampling Library"
  #   excerpt: "Turing provides **Hamiltonian Monte Carlo** (HMC) sampling for differentiable posterior distributions, **Particle MCMC** sampling for complex posterior distributions involving discrete variables and stochastic control flow, and **Gibbs** sampling which combines particle MCMC, HMC and many other MCMC algorithms."
  # - title: "Placeholder"
  #   excerpt: "Something cool about Turing"
  # - title: "Placeholder"
  #   excerpt: "Something else cool about Turing"

tomcat:
  - title: "Large Sampling Library"
    excerpt: "Turing provides Hamiltonian Monte Carlo (HMC) sampling for differentiable posterior distributions, Particle MCMC sampling for complex posterior distributions involving discrete variables and stochastic control flow, and Gibbs sampling which combines particle MCMC, HMC and many other MCMC algorithms."

citing:
  - title: "Citing Turing"
  - overlay_color: "#000"
  - excerpt: '<sub>If you use **Turing** for your own research, please consider citing the following publication: Hong Ge, Kai Xu, and Zoubin Ghahramani: **Turing: Composable inference for probabilistic programming.** AISTATS 2018 [pdf](http://proceedings.mlr.press/v84/ge18b.html) [bibtex](https://dblp.org/rec/bib2/conf/aistats/GeXG18.bib)</sub>'
---

{% include feature_row id="intro" type="left-center" %}

<!-- {% include feature_row id="current-features" type="center" %} -->

{% include feature_row type = "center"%}
{% include feature_row id="main-feature_row" type = "center"%}
{% include feature_row id="tomcat" type = "center"%}


{% include feature_row id="citing" type = "left-center" %}
