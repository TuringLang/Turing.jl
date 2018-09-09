
# Probabilistic Programming with Turing

## Introduction
This notebook is the first of a series of tutorials on the universal probabilistic programming language Turing.

**Turing** is probabilistic programming system written entirely in Julia. It has an intuitive modelling syntax and supports a wide range of sampling-based inference algorithms. Most importantly, Turing inference is composable: it combines Markov chain sampling operations on subsets of model variables, e.g. using a combination of a Hamiltonian Monte Carlo (HMC) engine and a particle Gibbs (PG) engine. This composable inference engine allows the user to easily switch between black-box style inference methods such as HMC and customized inference methods.

Familiarity with Julia is assumed through out this notebook. If you are new to Julia, [Learning Julia](https://julialang.org/learning/) is a good starting point.

For users new to Bayesian machine learning, please consider more thorough introductions to the field, such as [Pattern Recognition and Machine Learning](https://www.springer.com/us/book/9780387310732). This notebook tries to provide an intuition for Bayesian inference and gives a simple example on how to use Turing. Note that this notebook is not a comprehensive introduction to Bayesian machine learning.

## Coin Flipping without Turing
The following example aims to illustrate the effect of updating our beliefs with every piece of new evidence we observe. In particular, we will assume that we are unsure about the probability of heads in a coin flip. To get an intuitive understanding of what "updating our beliefs" is, we will visualize the probability of heads in a coin flip after each observed evidence.

First, let's load some of the packages we are going to need to flip a coin (`Random`, `Distributions`) and show our results (`Plots`). You will note that **Turing** is not an import here -- we are not going to need it for this example. If you are already familiar with posterior updates, you can proceed to the next step.

```julia
# using Base modules
using Random

# load a plotting library
using Plots

# load the distributions library
using Distributions
```

Next, we configure our posterior update model. First, let's set the true probability that any coin flip will turn up heads and set the number of coin flips we will show our model:

```julia
# set the true probability of heads in a coin
p_true = 0.5

# iterate from having seen 0 observations to 100 observations
Ns = 0:100;
```

We will use the Bernoulli distribution to flip 100 coins, and collect the results in a variable called `data`.

```julia
# draw data from a Bernoulli distribution, i.e. draw heads or tails
Random.seed!(12)
data = rand(Bernoulli(p_true), last(Ns))

# here's what the first five coin flips look like:
data[1:5]
```

After flipping all our coins, we want to set a prior belief about what we think the distribution of coinflips look like. In our case, we are going to choose a common prior distribution called the [Beta](https://en.wikipedia.org/wiki/Beta_distribution) distribution. We will allow this distribution to change as we let our model see more evidence of coin flips.

```julia
# our prior belief about the probability of heads in a coin toss
prior_belief = Beta(1, 1);
```

With our priors set and our data at hand, we can finally run our simple posterior update model.

This is a fairly simple process. We expose one additional coin flip to our model every iteratior, such that the first run only sees the first coin flip, while the last iteration sees all the coin flips. Then, we set the `updated_belief` variable to an updated version of the original Beta distribution after accounting for the new proportion of heads and tails.

```julia
# this is required for plotting only
x = range(0, stop = 1, length = 100)

# make an animation
animation = @animate for (i, N) in enumerate(Ns)

    # count the number of heads and tails
    heads = sum(data[1:i-1])
    tails = N - heads

    # update our prior belief in closed form (this is possible because we use a conjugate prior)
    updated_belief = Beta(prior_belief.α + heads, prior_belief.β + tails)

    # plotting
    plot(x, pdf.(Ref(updated_belief), x),
        size = (500, 250),
        title = "Updated belief after $N observations",
        xlabel = "probability of heads",
        ylabel = "",
        legend = nothing,
        xlim = (0,1),
        fill=0, α=0.3, w=3)
    vline!([p_true])
end;
```

![animation](https://user-images.githubusercontent.com/7974003/44995702-37c1b200-af9c-11e8-8b26-c88a528956af.gif)

The animation above shows that with increasing evidence our belief about the probability of heads in a coin flip slowly adjusts towards the true value. The orange line in the animation represents the true probability of seeing heads on a single coin flip, while the mode of the distribution shows what the model believes the probability of a heads is given the evidence it has seen.

## Coin Flipping with Turing

In the previous example, we used the fact that our prior distribution is a [conjugate prior](https://en.wikipedia.org/wiki/Conjugate_prior). Note that a closed-form expression (the `updated_belief` expression) for the posterior is not accessible in general and usually does not exist for more interesting models.

We are now going to move away from the closed-form expression above and specify the same model using Turing*. To do so, we will first need to import `Turing`, `MCMCChain`, `Distributions`, and `StatPlots`. `MCMChain` is a library built by the Turing team to help summarize Markov Chain Monte Carlo (MCMC) simulations, as well as a variety of utility functions for diagnostics and visualizations.

```julia
# load Turing and MCMCChain
using Turing, MCMCChain

# load the distributions library
using Distributions

# load stats plots for density plots
using StatPlots
```

First, we will define the coin-flip model using Turing.

```julia
@model coinflip(y) = begin

    # our prior belief about the probability of heads in a coin
    p ~ Beta(1, 1)

    # the number of observations
    N = length(y)
    for n in 1:N
        # heads or tails of a coin are drawn from a Bernoulli distribution
        y[n] ~ Bernoulli(p)
    end
end;
```

After defining the model, we can approximate the posterior distribution by pulling samples from the distribution. In this example, we use a [Hamiltonian Monte Carlo](https://en.wikipedia.org/wiki/Hamiltonian_Monte_Carlo) sampler to construct these samples. Later tutorials will give more information on the samplers available in Turing and discuss their use for different models.

```julia
# setting of Hamiltonian Monte Carlo (HMC) sampler
iterations = 1000
ϵ = 0.05
τ = 10

# start sampling
chain = sample(coinflip(data), HMC(iterations, ϵ, τ));
```

After finishing the sampling process, we can visualize the posterior distribution approximated using Turing against the posterior distribution in closed-form. We can extract the chain data from the sampler using the `Chains(chain[:p])` function. This contains all the values of `p` we drew while sampling.

```julia
# construct summary of the sampling process for the parameter p, i.e. the probability of heads in a coin
p_summary = Chains(chain[:p])
```

```
Object of type "Chains"

Iterations = 1:1000
Thinning interval = 1
Chains = 1
Samples per chain = 1000

[0.859911; 0.219831; … ; 0.496273; 0.473286]
```

Now we can build our plot:

```julia
# compute the posterior distribution in closed-form
N = length(data)
heads = sum(data)
updated_belief = Beta(prior_belief.α + heads, prior_belief.β + N - heads)

# visualize a blue density plot of the approximate posterior distribution using HMC (see Chain 1 in the legend)
p = densityplot(p_summary, xlim = (0,1), legend = :best, w = 2, c = :blue)

# visualize a green density plot of posterior distribution in closed-form
plot!(p, range(0, stop = 1, length = 100), pdf.(Ref(updated_belief), range(0, stop = 1, length = 100)),
        xlabel = "probability of heads", ylabel = "", title = "", xlim = (0,1), label = "Closed-form",
        fill=0, α=0.3, w=3, c = :lightgreen)

# visualize the true probability of heads in red
vline!(p, [p_true], label = "True probability", c = :red);
```

![sdf](https://user-images.githubusercontent.com/7974003/44995682-25477880-af9c-11e8-850b-36e4b6d756ea.png)

As we can see, the Turing model closely approximates the true probability. Hopefully this has provided an introduction to Turing's simpler applications. More advanced usage is demonstrated in later tutorials.
