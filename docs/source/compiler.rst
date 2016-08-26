The Turing Language
=========

Turing automates Bayesian inference by firstly defining a probabilistic model as a Julia program and then running this program using universal inference engines. This program is basically a normal Julia program extended with three probabilistic operations, namely ``@assume``, ``@observe`` and ``@predict``, wrapped in a ``@model`` scope.

Three probabilistic operations, ``@assume``, ``@observe`` and ``@predict``, are responsible for defining the prior probability, the likelihood and the priors to be outputted by the sampler respectively.

In detail, these three operations are supported with the syntax below.

* ``@assume x ~ D``: declare that the (prior) variable ``x`` is drawn from the distribution ``D``.

  Note: ``x`` will either be drawn from the distribution or be set using the current value stored in the sampler.

* ``@observe y ~ D``: declare that the value ``y`` is observed to be drawn from the distribution ``D``.

  Note: ``y`` is ought to have a concrete value in the current scope of the program.

* ``@predict x``: declare that which prior(s) declared by ``@assume`` (e.g. ``x`` here) should be output from the inference engine.

Distributions here are declared in form of standard mathematical form, e.g. ``Normal(0, 1)`` or ``Bernoulli(0.33)``. The Julia package ``Distributions`` supports most of the common distributions, however, due to the fact the distributions in the package are not differentiable, a wrapper of common distributions are implemented inside Turing.

Additionally, variables here can be annotated with additional arguments, ``static`` and ``param``, passed in the distribution, e.g. ``@assume mu ~ Normal(0, 1; static=true)``, to indicate their properties. In specific, these annotations are aimed to be used as below.

* ``static``: if this argument is set, it means that the existence of the corresponding variable does not rely on other variables. Therefore the variables exists in each execution of the program.

  Note: When this argument is set to ``true`` and the distribution is differentiable, the corresponding variable could be efficiently sampled by samplers like HMC.

* ``param``: if this argument is set, it means that the corresponding variable can be treated as a model parameter.

  Note: When this argument is set to ``true``, the corresponding variable could be efficiently sampled by samplers like SMC2.

This annotation feature is aimed to support a future Gibbs sampler, which combines PG and HMC by sampling discrete variables by PG and continuous variables by HMC respectively.
