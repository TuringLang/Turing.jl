var store = [{
        "title": "Advanced Usage",
        "excerpt":"How to Define a Customized Distribution Turing.jl supports the use of distributions from the Distributions.jl package. By extension it also supports the use of customized distributions, by defining them as subtypes of Distribution type of the Distributions.jl package, as well as corresponding functions. Below shows a workflow of how to...","categories": [],
        "tags": [],
        "url": "http://localhost:4000/docs/advanced/",
        "teaser":null},{
        "title": "Library",
        "excerpt":"Modelling # Turing.Core.@model — Macro. @model(body)Macro to specify a probabilistic model. Example: Model definition: @model model_generator(x = default_x, y) = begin ...endExpanded model definition # Allows passing arguments as kwargsmodel_generator(; x = nothing, y = nothing)) = model_generator(x, y)function model_generator(x = nothing, y = nothing) pvars, dvars = Turing.get_vars(Tuple{:x, :y},...","categories": [],
        "tags": [],
        "url": "http://localhost:4000/docs/library/",
        "teaser":null},{
        "title": "Automatic Differentiation",
        "excerpt":"Switching AD Modes Turing supports two types of automatic differentiation (AD) in the back end during sampling. The current default AD mode is ForwardDiff, but Turing also supports Tracker-based differentation. To switch between ForwardDiff and Tracker, one can call function Turing.setadbackend(backend_sym), where backend_sym can be :forward_diff or :reverse_diff. Compositional Sampling...","categories": [],
        "tags": [],
        "url": "http://localhost:4000/docs/autodiff/",
        "teaser":null},{
        "title": "Contributing",
        "excerpt":"Turing is an open source project. If you feel that you have some relevant skills and are interested in contributing, then please do get in touch. You can contribute by opening issues on GitHub or implementing things yourself and making a pull request. We would also appreciate example models written...","categories": [],
        "tags": [],
        "url": "http://localhost:4000/docs/contributing/",
        "teaser":null},{
        "title": "Style Guide",
        "excerpt":"This style guide is adapted from Invenia’s style guide. We would like to thank them for allowing us to access and use it. Please don’t let not having read it stop you from contributing to Turing! No one will be annoyed if you open a PR whose style doesn’t follow...","categories": [],
        "tags": [],
        "url": "http://localhost:4000/docs/style-guide/",
        "teaser":null},{
        "title": "Using DynamicHMC",
        "excerpt":"Turing supports the use of DynamicHMC as a sampler through the use of the DynamicNUTS function. This is a faster version of Turing’s native NUTS implementation. DynamicNUTS is not appropriate for use in compositional inference. If you intend to use Gibbs sampling, you must use Turing’s native NUTS function. To...","categories": [],
        "tags": [],
        "url": "http://localhost:4000/docs/dynamichmc/",
        "teaser":null},{
        "title": "Getting Started",
        "excerpt":"Installation To use Turing, you need to install Julia first and then install Turing. Install Julia You will need to install Julia 1.0 or greater, which you can get from the official Julia website. Install Turing.jl Turing is an officially registered Julia package, so the following will install a stable...","categories": [],
        "tags": [],
        "url": "http://localhost:4000/docs/get-started/",
        "teaser":null},{
        "title": "Guide",
        "excerpt":"Basics Introduction A probabilistic program is Julia code wrapped in a @model macro. It can use arbitrary Julia code, but to ensure correctness of inference it should not have external effects or modify global state. Stack-allocated variables are safe, but mutable heap-allocated objects may lead to subtle bugs when using...","categories": [],
        "tags": [],
        "url": "http://localhost:4000/docs/guide/",
        "teaser":null},{
        "title": "Turing Documentation",
        "excerpt":"Welcome to the documentation for Turing 0.6.4. Introduction Turing is a universal probabilistic programming language with an intuitive modelling interface, composable probabilistic inference and computational scalability. Turing provides Hamiltonian Monte Carlo (HMC) and particle MCMC sampling algorithms for complex posterior distributions (e.g. those involving discrete variables and stochastic control flows)....","categories": [],
        "tags": [],
        "url": "http://localhost:4000/docs/",
        "teaser":null},{
        "title": "Probablistic Programming in Thirty Seconds",
        "excerpt":"If you are already well-versed in probabalistic programming and just want to take a quick look at how Turing’s syntax works or otherwise just want a model to start with, we have provided a Bayesian coin-flipping model to play with. This example can be run on however you have Julia...","categories": [],
        "tags": [],
        "url": "http://localhost:4000/docs/quick-start/",
        "teaser":null},{
        "title": "Sampler Visualization",
        "excerpt":"## Introduction## The CodeFor each sampler, we will use the same code to plot sampler paths. The block below loads the relevant libraries and defines a function for plotting the sampler's trajectory across the posterior.The Turing model definition used here is not especially practical, but it is designed in such...","categories": [],
        "tags": [],
        "url": "http://localhost:4000/docs/sampler-viz/",
        "teaser":null},{
        "title": "Sampler Visualization",
        "excerpt":"Introduction The Code For each sampler, we will use the same code to plot sampler paths. The block below loads the relevant libraries and defines a function for plotting the sampler’s trajectory across the posterior. The Turing model definition used here is not especially practical, but it is designed in...","categories": [],
        "tags": [],
        "url": "http://localhost:4000/docs/sampler-viz/",
        "teaser":null},{
        "title": "Introduction to Turing",
        "excerpt":"Introduction This is the first of a series of tutorials on the universal probabilistic programming language Turing. Turing is probabilistic programming system written entirely in Julia. It has an intuitive modelling syntax and supports a wide range of sampling-based inference algorithms. Most importantly, Turing inference is composable: it combines Markov...","categories": [],
        "tags": [],
        "url": "http://localhost:4000/tutorials/0-introduction/",
        "teaser":null},{
        "title": "Unsupervised Learning using Bayesian Mixture Models",
        "excerpt":"The following tutorial illustrates the use Turing for clustering data using a Bayesian mixture model. The aim of this task is to infer a latent grouping (hidden structure) from unlabelled data. More specifically, we are interested in discovering the grouping illustrated in figure below. This example consists of 2-D data...","categories": [],
        "tags": [],
        "url": "http://localhost:4000/tutorials/1-gaussianmixturemodel/",
        "teaser":null},{
        "title": "Bayesian Logistic Regression",
        "excerpt":"Bayesian logistic regression is the Bayesian counterpart to a common tool in machine learning, logistic regression. The goal of logistic regression is to predict a one or a zero for a given training item. An example might be predicting whether someone is sick or ill given their symptoms and personal...","categories": [],
        "tags": [],
        "url": "http://localhost:4000/tutorials/2-logisticregression/",
        "teaser":null},{
        "title": "Bayesian Neural Networks",
        "excerpt":"In this tutorial, we demonstrate how one can implement a Bayesian Neural Network using a combination of Turing and Flux, a suite of tools machine learning. We will use Flux to specify the neural network’s layers and Turing to implement the probabalistic inference, with the goal of implementing a classification...","categories": [],
        "tags": [],
        "url": "http://localhost:4000/tutorials/3-bayesnn/",
        "teaser":null},{
        "title": "Bayesian Hidden Markov Models",
        "excerpt":"This tutorial illustrates training Bayesian Hidden Markov Models (HMM) using Turing. The main goals are learning the transition matrix, emission parameter, and hidden states. For a more rigorous academic overview on Hidden Markov Models, see An introduction to Hidden Markov Models and Bayesian Networks (Ghahramani, 2001). Let’s load the libraries...","categories": [],
        "tags": [],
        "url": "http://localhost:4000/tutorials/4-bayeshmm/",
        "teaser":null},{
        "title": "Probabilistic Modelling using the Infinite Mixture Model",
        "excerpt":"In many applications it is desirable to allow the model to adjust its complexity to the amount the data. Consider for example the task of assigning objects into clusters or groups. This task often involves the specification of the number of groups. However, often times it is not known beforehand...","categories": [],
        "tags": [],
        "url": "http://localhost:4000/tutorials/6-infinitemixturemodel/",
        "teaser":null},{
        "title": "Tutorials",
        "excerpt":"This section contains tutorials on how to implement common models in Turing. If you prefer to have an interactive Jupyter notebook, please fork or download the TuringTutorials repository. A list of all the tutorials available can be found to the left. The introduction tutorial contains an introduction to coin flipping...","categories": [],
        "tags": [],
        "url": "http://localhost:4000/tutorials/",
        "teaser":null}]
