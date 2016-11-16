Get Started
===========

Introduction
------------

Turing is a Julia library for probabilistic programming. A Turing probabilistic program is just a normal Julia program, wrapped in a ``@model`` macro, that uses some of the special macros listed below. Available inference methods include Importance Sampling (IS), Sequential Monte Carlo (SMC), Particle Gibbs (PG) and Hamiltonian Monte Carlo (HMC).

Authors: `Hong Ge <http://mlg.eng.cam.ac.uk/hong/>`__, `Adam
Scibior <http://mlg.eng.cam.ac.uk/?portfolio=adam-scibior>`__, `Matej
Balog <http://mlg.eng.cam.ac.uk/?portfolio=matej-balog>`__, `Zoubin
Ghahramani <http://mlg.eng.cam.ac.uk/zoubin/>`__

Relevant papers
~~~~~~~~~~~~~~~

1. Ghahramani, Zoubin. "Probabilistic machine learning and artificial intelligence." Nature 521, no. 7553 (2015): 452-459. (`pdf <http://www.nature.com/nature/journal/v521/n7553/full/nature14541.html>`__)
2. Ge, Hong, Adam Scibior, and Zoubin Ghahramani. "Turing: A fast imperative probabilistic programming language." (In submission).
3. Ge, Hong and Ścibior, Adam and Xu, Kai and Ghahramani, Zoubin. "Turing: A fast imperative probabilistic programming language."

Example
~~~~~~~

Below is a simple Gaussian model with unknown mean and variance.

.. code:: julia

    @model gaussdemo begin
      @assume s ~ InverseGamma(2,3)
      @assume m ~ Normal(0,sqrt(s))
      @observe 1.5 ~ Normal(m, sqrt(s))
      @observe 2.0 ~ Normal(m, sqrt(s))
      @predict s m
    end

Installation
------------

To use Turing, you need install Julia first and then install Turing.

Julia
~~~~~

You will need Julia 0.5 (or 0.4; but 0.5 is recommended), which you can get from `the official Julia website <http://julialang.org/downloads/>`_.

It provides three options for users

1. A command line version (`Julia/downloads <http://julialang.org/downloads/>`_)
2. A community maintained IDE `Juno<http://www.junolab.org/>`_
3. `JuliaBox.com<https://www.juliabox.com/>`_ - a Jupyter notebook in the browser

For command line version, we recommend that you install a pre-compiled package, as Turing may not work correctly with Julia built form source.

Juno also needs the command line version installed. This IDE is recommended for heavy users who require features like debugging, quick documentation check, etc.

JuliaBox is a free-installed Jupyter notebook for Julia. You can follow the following section to give a shot to Turing without installing Julia on your machine in few seconds.

  **IMPORTANT**: as JuliaBox is still in beta version, it has a bug which will cause Turing fail to build from Jupyter. For installation of Turing on JuliaBox, please go to ``Console`` tab (from the navigation bar on the top), run ``PATH=/opt/julia-0.5.0/bin:$PATH; export PATH`` and install Turing from terminal (see next section).

  **TIP**: you can copy the command from here and paste it into JuliaBox by right click on the virtual terminal and choose paste from browser.

Turing
~~~~~~

Turing is an officially registered Julia package, so the following should install a stable version of Turing:

.. code:: julia

    Pkg.update()
    Pkg.add("Turing")
    Pkg.build("Turing")
    Pkg.test("Turing")

If you want to use the latest version of Turing with some experimental samplers, you can try the following instead:

.. code:: julia

    Pkg.update()
    Pkg.clone("Turing")
    Pkg.build("Turing")
    Pkg.test("Turing")

If all tests pass, you're ready to start using Turing.

A Short Tutorial
----------------

Below is a short but necessary introduction to our APIs and how to use them. For details, please refer to :ref:`APIs` and :ref:`Development Notes`.

Modelling API
~~~~~~~~~~~~~

A probabilistic program is Julia code wrapped in a ``@model`` macro. It can use arbitrary Julia code, but to ensure correctness of inference it should not have external effects or modify global state. Stack-allocated
variables are safe, but mutable heap-allocated objects may lead to subtle bugs when using task copying. To help avoid those we provide a Turing-safe datatype ``TArray`` that can be used to create mutable arrays in Turing programs.

For probabilistic effects, Turing programs should use the following macros:

``@assume x ~ distr`` where ``x`` is a symbol and ``distr`` is a distribution. Inside the probabilistic program this puts a random variable named ``x``, distributed according to ``distr``, in the current
scope. ``distr`` can be a value of any type that implements ``rand(distr)``, which samples a value from the distribution ``distr``.

``@observe y ~ distr`` This is used for conditioning in a style similar to Anglican. Here ``y`` should be a value that is observed to have been drawn from the distribution ``distr``. The likelihood is computed using
``pdf(distr,y)`` and should always be positive to ensure correctness of inference algorithms. The observe statements should be arranged so that every possible run traverses all of them in exactly the same order. This
is equivalent to demanding that they are not placed inside stochastic control flow.

``@predict x`` Registers the current value of ``x`` to be inspected in the results of inference.

Inference API
~~~~~~~~~~~~~

Inference methods are functions which take the probabilistic program as one of the arguments.

.. code:: julia

    #  Run sampler, collect results
    chain = sample(gaussdemo, SMC(500))
    chain = sample(gaussdemo, PG(10,500))
    chain = sample(gaussdemo, HMC(1000, 0.1, 5))

The arguments for each sampler are

* SMC: number of particles
* PG: number of praticles, number of iterations
* HMC: number of samples, leapfrog step size, leapfrog step numbers

Task copying
~~~~~~~~~~~~

Turing `copies <https://github.com/JuliaLang/julia/issues/4085>`__ Julia
tasks to deliver efficient inference algorithms, but it also provides
alternative slower implementation as a fallback. Task copying is enabled
by default. Task copying requires building a small C program, which
should be done automatically on Linux and Mac systems that have GCC and
Make installed.
