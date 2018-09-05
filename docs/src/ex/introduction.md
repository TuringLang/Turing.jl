
# Probabilistic Programming with Turing

---

## Introduction
This notebook is the first of a series of tutorials on the universal probabilistic programming language **Turing**.

**Turing** is probabilistic programming system written entirely in *Julia*. It has an intuitive modelling syntax and supports a wide range of sampling-based inference algorithms. Most importantly, **Turing** inference is composable: it combines Markov chain sampling operations on subsets of model variables, e.g. using a combination of a Hamiltonian Monte Carlo (HMC) engine and a particle Gibbs (PG) engine. This composable inference engine allows the user to easily switch between black-box style inference methods such as HMC and customized inference methods.

---
Familiarity with Julia is assumed through out this notebook. If you are new to Julia, [Learning Julia](https://julialang.org/learning/) is a good starting point.

For users new to Bayesian machine learning we refer to further resources, e.g. the Pattern Recognition and Machine Learning book. This notebook tries to provide an intuition for Bayesian inference and gives a simple example on how to use **Turing**. Note that this notebook is not a comprehensive introduction to Bayesian machine learning.

## Example: Coin-Flipping  (Julia 1.0)
The following example aims to illustrate the effect of updating our beliefs with each new evidence we observe. In particular, we will assume that we are unsure about the probability of heads in a coin flip. To get an intuitive understanding of what "updating our beliefs" is, we will visualize the probability of heads in a coin flip after each observed evidence.

---
Note that we will not need **Turing** for this particular example. If you are familiar with posterior updates, feel free to proceed to the next tutorial.


```julia
# using Base modules
using Random

# load a plotting library
using Plots

# use the PyPlot backend for plotting (this is optional)
#pyplot();

# load the distributions library
using Distributions
```


```julia
# set the true probability of heads in a coin
p_true = 0.5

# iterate from having seen 0 observations to 100 observations
Ns = 0:100

# draw data from a Bernoulli distribution, i.e. draw heads or tails
Random.seed!(12)
data = rand(Bernoulli(p_true), last(Ns))

# our prior belief about the probability of heads in a coin
prior_belief = Beta(1, 1)

# this is required for plotting only
x = range(0, stop = 1, length = 100)

# make an animation
animation = @animate for (i, N) in enumerate(Ns)

    # count the number of heads
    heads = sum(data[1:i-1])

    # update our prior belief in closed form (this is possible because we use a conjugate prior)
    updated_belief = Beta(prior_belief.α + heads, prior_belief.β + N - heads)

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

The animation above shows that with increasing evidence our belief about the probability of heads in a coin flip turns towards the true value (compare the mode of the beta distribution to the orange line).

### Example: Coin-Flipping - part 2  (Julia 1.0)

In the previous example, we used the fact that our prior distribution is a conjugate prior. Note that a closed-form expression for the posterior is not accessible in general and usually does not exist for more interesting models.


```julia
# load Turing and MCMCChain
using Turing, MCMCChain

# load stats plots for density plots
using StatPlots
```

In the following, we will compare the result of Turing against the posterior distribution obtained in closed-form.

---

At first, we will define the coin-flip model using Turing.


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

After defining the model, we can approximate the posterior distribution using samples from the distribution. In this example, we use a Hamiltonian Monte Carlo sampler to construct these samples. Later tutorials will give more information on the samplers available in Turing and discuss their use for different models.


```julia
# setting of Hamiltonian Monte Carlo (HMC) sampler
iterations = 1000
ϵ = 0.05
τ = 10

# start sampling
chain = sample(coinflip(data), HMC(iterations, ϵ, τ));
```

---

After finishing the sampling process, we can visualize the posterior distribution approximated using Turing against the posterior distribution in closed-form.


```julia
# construct summary of the sampling process for the parameter p, i.e. the probability of heads in a coin
p_summary = Chains(chain[:p])

# compute the posterior distribution in closed-form
N = length(data)
heads = sum(data)
updated_belief = Beta(prior_belief.α + heads, prior_belief.β + N - heads)

# visualize a blue density plot of the approximate posterior distribution using HMC (see Chain 1 in the legend)
p = densityplot(p_summary, xlim = (0,1), legend = :best, w = 2, c = :blue)

# visualize a green density plot of posterior distribution in closed-form
plot!(p, range(0, stop = 1, length = 100), pdf.(Ref(updated_belief), range(0, stop = 1, length = 100)),
        xlabel = "probability of heads", ylabel = "", title = "", xlim = (0,1), label = "closed-form",
        fill=0, α=0.3, w=3, c = :lightgreen)

# visualize the true probability of heads in red
vline!(p, [p_true], label = "true probability", c = :red)
```

![sdf](https://user-images.githubusercontent.com/7974003/44995682-25477880-af9c-11e8-850b-36e4b6d756ea.png)
