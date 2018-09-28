---
layout: turing
title: "The Turing language for probabilistic machine learning"
---

**Turing** is a *universal* probabilistic programming language with an intuitive modelling interface, composable probabilistic inference and computational scalability.

Turing provides **Hamiltonian Monte Carlo** (HMC) and **particle MCMC** sampling algorithms for complex posterior distributions (e.g. those involving discrete variables and stochastic control flows). Current features include:

- **Universal** probabilistic programming with an intuitive modelling interface;
- **Hamiltonian Monte Carlo** (HMC) sampling for differentiable posterior distributions;
- **Particle MCMC** sampling for complex posterior distributions involving discrete variables and stochastic control flow; and
- **Gibbs** sampling that combines particle MCMC,  HMC and many other MCMC algorithms.

# Resources

Please visit the [Turing.jl documentation](http://turing.ml/latest) for documentation, tutorials (e.g. [get started](http://turing.ml/latest/get-started.html)) and other topics (e.g. [advanced usages](http://turing.ml/latest/advanced.html)). Below are some example models for Turing.

- [Introduction](http://turing.ml/latest/ex/0_Introduction.html)
- [Gaussian Mixture Model](https://github.com/TuringLang/TuringTutorials/tree/master/old-notebooks/GMM.ipynb)
- [Bayesian Hidden Markov Model](https://github.com/TuringLang/TuringTutorials/tree/master/old-notebooks/BayesHmm.ipynb)
- [Factorical Hidden Markov Model](https://github.com/TuringLang/TuringTutorials/tree/master/old-notebooks/FHMM.ipynb)
- [Topic Models: LDA and MoC](https://github.com/TuringLang/TuringTutorials/tree/master/old-notebooks/TopicModels.ipynb)

# Citing Turing

To cite Turing, please refer to the following paper. A sample BiBTeX entry entry is given below:

```
{% raw %}
@InProceedings{turing17,
  title = 	 {{T}uring: a language for flexible probabilistic inference},
  author = 	 {Ge, Hong and Xu, Kai and Ghahramani, Zoubin},
  booktitle = 	 {Proceedings of the 21th International Conference on Artificial Intelligence and Statistics},
  year = 	 {2018},
  series = 	 {Proceedings of Machine Learning Research},
  publisher = 	 {PMLR},
}
{% endraw %}
```

# Other probablistic/deep learning languages

- [Stan](http://mc-stan.org/)
- [Infer.NET](https://www.microsoft.com/en-us/research/project/infernet/)
- [PyTorch](http://pytorch.org/) / [Pyro](https://github.com/uber/pyro)
- [TensorFlow](https://www.tensorflow.org/) / [Edward](http://edwardlib.org/)
- [DyNet](https://github.com/clab/dynet)
