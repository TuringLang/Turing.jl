---
title: Primer on Bayesian Inference for Non-Experts
---

# Parametric models

Machine learning models describe interesting patterns and relationships in observed data. For example, a regression model can describe how one observed variable may be described as a function of the remaining observed variables. A clustering model can describe the membership of each data point to a specific cluster. Typically, machine learning models are parametric, where different values of their parameters make a different claim about the data. For example, the simple linear regression model `y = c*x` has a parameter `c`. `c = 1` claims that `y` and `x` are equal and `c = 0` claims that `y` is 0 regardless of the value of `x`. Given some data, one might want to find a reasonable value of the parameter `c` based on some reasonable criteria. In classical linear regression, a least square solution would be sought to minimize the sum of the error squared between the observed `y` values and their respective predicted `y` values given the respective observed `x` values for all the data points `(x, y)`. 

# Probabilistic models

Probabilistic machine learning models describe interesting patterns and relationships in observed data in the language of probabilities. For instance, instead of writing `y = c*x`, one would write `y|x ~ Normal(c*x, σ)` which claims that `y` is normally distributed with a mean `c*x` and standard deviation `σ`. `c` and `σ` are the parameters of this model. Now given some data, one might want to find reasonable values for both `c` and `σ`. One way to choose such parameter values is to pick the parameters that maximize the probability of the observed data, also known as the likelihood. This is called maximum likelihood estimation (MLE).

Let `P` be the set of model parameters, e.g. scalar `c` and scalar `σ`. Let `D` by the set of data points, e.g. vector `X` and vector `Y` such that `Y[i]|X[i] ~ Normal(c*X[i], σ)`. `X` here is assumed to be deterministic, such that `p(X,Y) = p(Y|X)`. `p(D|P) = p(X,Y|c,σ) = p(Y|X,c,σ) = prod(pdf(Normal(c*X[i], σ), Y[i]) for i in 1:N)`, where `N = length(X)`. `p(D|P)` is known as the likelihood probability distribution. Taking the product of many numbers less than 1 is likely to cause a phenomenon known as underflow, where the product becomes smaller than the smallest number that can be represented in the machine precision used. For this reason, the `log` of the likelihood is often maximized instead. Maximizing the `log` of a function is equivalent to maximizing the function itself since `log` is a monotonically increasing function. `log p(D|P) = sum(logpdf(Normal(c*X[i], σ), Y[i]) for i in 1:N)`.

The result of MLE is a single value for each parameter that together make a probabilistic claim about how the data is generated. For instance, given a new value for `x`, one can sample an arbitrary number of `y` samples from the distribution `Normal(c*x, σ)`. It is for this reason that a probabilistic model is sometimes referred to as a generative model. In this case, the claim made is about `y` given `x`, assuming `x` is deterministic. However, another claim can be made about both `x` and `y` simultaneously using a different probabilistic model, assuming both are probabilistic.

# Bayesian inference

Instead of seeking a single value for each parameter for the probabilistic model `p(D|P)`, one might wonder what are all the likely values that could have generated the observed data? This question can be answered using Bayesian inference, also known as probabilistic inversion. 

The goal of Bayesian inference is to invert the probability distribution `p(D|P)` to obtain the probability distribution `p(P|D)` instead, given some prior distribution `p(P)` that resembles our bias or prior knowledge. `p(P|D)` is known as the posterior distribution. The prior distribution `p(P)` should describe our uncertainty about the values of the parameters `P` prior to observing any data. For example, let a coin flip generative model be modeled using a Bernoulli distribution, `head ~ Bernoulli(phead)`, where `phead` is the probability of getting an outcome `head = true`. The prior on `phead` can be `phead ~ Uniform(0, 1)`. This prior distribution says that we are not sure if the coin is biased or not and by how much. In other words, before observing any data we cannot say anything meaningful about the probability of getting a head, only that `phead` must between 0 and 1. One can also choose to use a `Beta` distribution for the prior to reflect a different state of uncertainty. The `Beta` distribution's support is also between 0 and 1. Therefore, we are saying that any value of `phead < 0` or `phead > 1` is impossible and should not be considered. For unbounded parameters, the choice of the prior distribution can be more difficult and requires some domain knowledge about the likely values of these parameters in the model.

Inverting the likelihood probability distribution `p(D|P)` to get the posterior distribution `p(P|D)` is done using Bayes' rule:
```
p(P|D) = p(P,D) / p(D) = p(P) * p(D|P) / p(D)
``` 
where `p(P,D) = p(P) * p(D|P)` is the joint probability distribution of `P` and `D`, and `p(D) = ∫ p(P,D) dP` is the normalization constant that marginalizes out `P` in `p(P,D)` given the data observed. `p(D)` is a constant in `P` since we integrate `P` out and is therefore only a function of the observed data `D`. Given a fixed data set `D`, the normalization constant will also be fixed.

In some cases, given the prior `p(P)` and the likelihood `p(D|P)`, one can successfully determine the normalization constant `p(D) = ∫ p(P,D) dP` by analytically performing the integration. For example, this is possible when the prior and likelihood are both Gaussian distributions. One can also use Gaussian quadrature integration to numerically evaluate `p(D)` for low dimensional `P`. However, in many cases, evaluating `p(D)` is analytically intractable, and for a high dimensional `P` the integration by Gaussian quadrature also becomes computationally intractable.

# Approximate Bayesian inference

When neither analytical nor numerical integration is tractable, instead of doing closed-form Bayesian inference, we can attempt to do approximate Bayesian inference. In approximate Bayesian inference, instead of finding a closed form solution for the posterior `p(P|D)`, we attempt to sample from the posterior distribution only using the joint probability distribution `p(P,D)`. `p(P,D)` is readily available from the prior and likelihood distributions. A family of algorithms were developed to sample from the posterior distribution `p(P|D) ∝ p(P,D)` only using the joint distribution `p(P,D)` without the normalization constant `p(D)`. This family of algorithms is known as Markov Chain Monte Carlo (MCMC). The goal of MCMC is to ensure that the sample we get at the end approximates an identically and independently distributed (iid) sample from the exact posterior distribution. Variational inference (VI) is another closely related family of algorithms for approximate Bayesian inference. 

# Turing.jl

`Turing` provides an easy syntax to define your probabilistic model and then run MCMC or VI on it using robust, readily available algorithms.

# More References

- [Probabilistic Programming and Bayesian Methods for Hackers](https://github.com/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers)
- Doing Bayesian Data Analysis by John Kruschke
- Statistical Rethinking A Bayesian Course with Examples in R and Stan by Richard McElreath
