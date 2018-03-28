---
layout: turing
title: "The Turing language for probabilistic machine learning"
---

**Turing** is a *universal* probabilistic programming language with a focus on intuitive modelling interface, composable probabilistic inference and computational scalability.

# What you get...

Turing provides **Hamiltonian Monte Carlo** (HMC) and **particle MCMC** sampling algorithms for complex posterior distributions (e.g. those involving discrete variables and stochastic control flows). Current features include:

- **Universal** probabilistic programming with an intuitive modelling interface
- **Hamiltonian Monte Carlo** (HMC) sampling for differentiable posterior distributions
- **Particle MCMC** sampling for complex posterior distributions involving discrete variables and stochastic control flows
- **Gibbs** sampling that combines particle MCMC,  HMC and many other MCMC algorithms

# Resources

Please visit [Turing.jl wiki](https://github.com/yebai/Turing.jl/wiki) for documentation, tutorials (e.g. [get started](https://github.com/yebai/Turing.jl/wiki/Get-started)) and other topics (e.g. [advanced usages](https://github.com/yebai/Turing.jl/wiki/Advanced-usages)). Below are some example models for Turing.

- [Introduction](https://nbviewer.jupyter.org/github/yebai/Turing.jl/blob/master/example-models/notebooks/Introduction.ipynb)
- [Gaussian Mixture Model](https://nbviewer.jupyter.org/github/yebai/Turing.jl/blob/master/example-models/notebooks/GMM.ipynb)
- [Bayesian Hidden Markov Model](https://nbviewer.jupyter.org/github/yebai/Turing.jl/blob/master/example-models/notebooks/BayesHmm.ipynb)
- [Factorical Hidden Markov Model](https://nbviewer.jupyter.org/github/yebai/Turing.jl/blob/master/example-models/notebooks/FHMM.ipynb)
- [Topic Models: LDA and MoC](https://nbviewer.jupyter.org/github/yebai/Turing.jl/blob/master/example-models/notebooks/TopicModels.ipynb)

# Citing Turing

To cite Turing, please refer to the following paper. Sample BibTeX entry is given below:

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

