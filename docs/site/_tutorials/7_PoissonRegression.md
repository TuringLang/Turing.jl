---
title: Bayesian Poisson Regression
permalink: /:collection/:name/
---

This notebook is ported from the [example notebook](https://docs.pymc.io/notebooks/GLM-poisson-regression.html) of PyMC3 on Poisson Regression.  

[Poisson Regression](https://en.wikipedia.org/wiki/Poisson_regression) is a technique commonly used to model count data. Some of the applications include predicting the number of people defaulting on their loans or the number of cars running on a highway on a given day. This example describes a method to implement the Bayesian version of this technique using Turing.

We will generate the dataset that we will be working on which describes the relationship between number of times a person sneezes during the day with his alcohol consumption and medicinal intake.

We start by importing the required libraries.

````julia
#Import Turing, Distributions and DataFrames
using Turing, Distributions, DataFrames, Distributed

# Import MCMCChain, Plots, and StatsPlots for visualizations and diagnostics.
using MCMCChains, Plots, StatsPlots

# Set a seed for reproducibility.
using Random
Random.seed!(12);

# Turn off progress monitor.
Turing.turnprogress(false)
````


````
false
````




# Generating data
We start off by creating a toy dataset. We take the case of a person who takes medicine to prevent excessive sneezing. Alcohol consumption increases the rate of sneezing for that person. Thus, the two factors affecting the number of sneezes in a given day are alcohol consumption and whether the person has taken his medicine. Both these variable are taken as boolean valued while the number of sneezes will be a count valued variable. We also take into consideration that the interaction between the two boolean variables will affect the number of sneezes

5 random rows are printed from the generated data to get a gist of the data generated.

````julia
theta_noalcohol_meds = 1    # no alcohol, took medicine
theta_alcohol_meds = 3      # alcohol, took medicine
theta_noalcohol_nomeds = 6  # no alcohol, no medicine
theta_alcohol_nomeds = 36   # alcohol, no medicine

# no of samples for each of the above cases
q = 100

#Generate data from different Poisson distributions
noalcohol_meds = Poisson(theta_noalcohol_meds)
alcohol_meds = Poisson(theta_alcohol_meds)
noalcohol_nomeds = Poisson(theta_noalcohol_nomeds)
alcohol_nomeds = Poisson(theta_alcohol_nomeds)

nsneeze_data = vcat(rand(noalcohol_meds, q), rand(alcohol_meds, q), rand(noalcohol_nomeds, q), rand(alcohol_nomeds, q) )
alcohol_data = vcat(zeros(q), ones(q), zeros(q), ones(q) )
meds_data = vcat(zeros(q), zeros(q), ones(q), ones(q) )

df = DataFrame(nsneeze = nsneeze_data, alcohol_taken = alcohol_data, nomeds_taken = meds_data, product_alcohol_meds = meds_data.*alcohol_data)
df[sample(1:nrow(df), 5, replace = false), :]
````


````
5×4 DataFrame
│ Row │ nsneeze │ alcohol_taken │ nomeds_taken │ product_alcohol_meds │
│     │ Int64   │ Float64       │ Float64      │ Float64              │
├─────┼─────────┼───────────────┼──────────────┼──────────────────────┤
│ 1   │ 8       │ 0.0           │ 1.0          │ 0.0                  │
│ 2   │ 5       │ 1.0           │ 0.0          │ 0.0                  │
│ 3   │ 0       │ 0.0           │ 0.0          │ 0.0                  │
│ 4   │ 0       │ 0.0           │ 0.0          │ 0.0                  │
│ 5   │ 38      │ 1.0           │ 1.0          │ 1.0                  │
````




# Visualisation of the dataset
We plot the distribution of the number of sneezes for the 4 different cases taken above. As expected, the person sneezes the most when he has taken alcohol and not taken his medicine. He sneezes the least when he doesn't consume alcohol and takes his medicine.

````julia
#Data Plotting

p1 = Plots.histogram(df[(df[:alcohol_taken] .== 0) .& (df[:nomeds_taken] .== 0), 1], title = "no_alcohol+meds")  
p2 = Plots.histogram((df[(df[:alcohol_taken] .== 1) .& (df[:nomeds_taken] .== 0), 1]), title = "alcohol+meds")  
p3 = Plots.histogram((df[(df[:alcohol_taken] .== 0) .& (df[:nomeds_taken] .== 1), 1]), title = "no_alcohol+no_meds")  
p4 = Plots.histogram((df[(df[:alcohol_taken] .== 1) .& (df[:nomeds_taken] .== 1), 1]), title = "alcohol+no_meds")  
plot(p1, p2, p3, p4, layout = (2, 2), legend = false)
````


![](/tutorials/figures/7_PoissonRegression_3_1.png)


We must convert our `DataFrame` data into the `Matrix` form as the manipulations that we are about are designed to work with `Matrix` data. We also separate the features from the labels which will be later used by the Turing sampler to generate samples from the posterior.

````julia
# Convert the DataFrame object to matrices.
data = Matrix(df[[:alcohol_taken, :nomeds_taken, :product_alcohol_meds]])
data_labels = df[:nsneeze]
data
````


````
400×3 Array{Float64,2}:
 0.0  0.0  0.0
 0.0  0.0  0.0
 0.0  0.0  0.0
 0.0  0.0  0.0
 0.0  0.0  0.0
 0.0  0.0  0.0
 0.0  0.0  0.0
 0.0  0.0  0.0
 0.0  0.0  0.0
 0.0  0.0  0.0
 ⋮            
 1.0  1.0  1.0
 1.0  1.0  1.0
 1.0  1.0  1.0
 1.0  1.0  1.0
 1.0  1.0  1.0
 1.0  1.0  1.0
 1.0  1.0  1.0
 1.0  1.0  1.0
 1.0  1.0  1.0
````




We must recenter our data about 0 to help the Turing sampler in initialising the parameter estimates. So, normalising the data in each column by subtracting the mean and dividing by the standard deviation:

````julia
# # Rescale our matrices.
data = (data .- mean(data, dims=1)) ./ std(data, dims=1)
````


````
400×3 Array{Float64,2}:
 -0.998749  -0.998749  -0.576628
 -0.998749  -0.998749  -0.576628
 -0.998749  -0.998749  -0.576628
 -0.998749  -0.998749  -0.576628
 -0.998749  -0.998749  -0.576628
 -0.998749  -0.998749  -0.576628
 -0.998749  -0.998749  -0.576628
 -0.998749  -0.998749  -0.576628
 -0.998749  -0.998749  -0.576628
 -0.998749  -0.998749  -0.576628
  ⋮                             
  0.998749   0.998749   1.72988 
  0.998749   0.998749   1.72988 
  0.998749   0.998749   1.72988 
  0.998749   0.998749   1.72988 
  0.998749   0.998749   1.72988 
  0.998749   0.998749   1.72988 
  0.998749   0.998749   1.72988 
  0.998749   0.998749   1.72988 
  0.998749   0.998749   1.72988
````




# Declaring the Model: Poisson Regression
Our model, `poisson_regression` takes four arguments:

- `x` is our set of independent variables;
- `y` is the element we want to predict;
- `n` is the number of observations we have; and
- `σ²` is the standard deviation we want to assume for our priors.

Within the model, we create four coefficients (`b0`, `b1`, `b2`, and `b3`) and assign a prior of normally distributed with means of zero and standard deviations of `σ²`. We want to find values of these four coefficients to predict any given `y`. 

Intuitively, we can think of the coefficients as:

- `b1` is the coefficient which represents the effect of taking alcohol on the number of sneezes; 
- `b2` is the coefficient which represents the effect of taking in no medicines on the number of sneezes; 
- `b3` is the coefficient which represents the effect of interaction between taking alcohol and no medicine on the number of sneezes; 

The `for` block creates a variable `theta` which is the weighted combination of the input features. We have defined the priors on these weights above. We then observe the likelihood of calculating `theta` given the actual label, `y[i]`.

````julia
# Bayesian poisson regression (LR)
@model poisson_regression(x, y, n, σ²) = begin
    b0 ~ Normal(0, σ²)
    b1 ~ Normal(0, σ²)
    b2 ~ Normal(0, σ²)
    b3  ~ Normal(0, σ²)
    for i = 1:n
        theta = b0 + b1*x[i, 1] + b2*x[i,2] + b3*x[i,3]
        y[i] ~ Poisson(exp(theta))
    end
end;
````




# Sampling from the posterior
We use the `NUTS` sampler to sample values from the posterior. We run multiple chains using the `mapreduce` function to nullify the effect of a problematic chain. We then use the Gelman, Rubin, and Brooks Diagnostic to check the convergence of these multiple chains.

````julia
# This is temporary while the reverse differentiation backend is being improved.
Turing.setadbackend(:forward_diff)

# Retrieve the number of observations.
n, _ = size(data)

# Sample using NUTS.

num_chains = 4
chains = mapreduce(c -> sample(poisson_regression(data, data_labels, n, 10), NUTS(2500, 200, 0.65) ), chainscat, 1:num_chains);
````


````
[NUTS] Finished with
  Running time        = 37.545159871000024;
  #lf / sample        = 0.0;
  #evals / sample     = 0.0004;
  pre-cond. metric    = [1.0, 1.0, 1.0, 1.0].
[NUTS] Finished with
  Running time        = 365.7509047770001;
  #lf / sample        = 0.0;
  #evals / sample     = 0.0004;
  pre-cond. metric    = [1.0, 1.0, 1.0, 1.0].
[NUTS] Finished with
  Running time        = 35.77645984200004;
  #lf / sample        = 0.0;
  #evals / sample     = 0.0004;
  pre-cond. metric    = [1.0, 1.0, 1.0, 1.0].
[NUTS] Finished with
  Running time        = 453.8648650210001;
  #lf / sample        = 0.0;
  #evals / sample     = 0.0004;
  pre-cond. metric    = [1.0, 1.0, 1.0, 1.0].
````




# Viewing the Diagnostics 
We use the Gelman, Rubin, and Brooks Diagnostic to check whether our chains have converged. Note that we require multiple chains to use this diagnostic which analyses the difference between these multiple chains. 

We expect the chains to have converged. This is because we have taken sufficient number of iterations (1500) for the NUTS sampler. However, in case the test fails, then we will have to take a larger number of iterations, resulting in longer computation time.

````julia
gelmandiag(chains)
````


````
Gelman, Rubin, and Brooks Diagnostic

│ Row │ parameters │ PSRF    │ 97.5%   │
│     │ Symbol     │ Float64 │ Float64 │
├─────┼────────────┼─────────┼─────────┤
│ 1   │ b0         │ 1.03468 │ 1.03948 │
│ 2   │ b1         │ 1.05339 │ 1.06082 │
│ 3   │ b2         │ 1.06552 │ 1.07763 │
│ 4   │ b3         │ 1.0341  │ 1.04598 │
│ 5   │ lf_eps     │ 3.11179 │ 5.30209 │
````




From the above diagnostic, we can conclude that the chains have converged because the PSRF values of the coefficients are close to 1. 

So, we have obtained the posterior distributions of the parameters. We transform the coefficients and recover theta values by taking the exponent of the meaned values of the coefficients `b0`, `b1`, `b2` and `b3`. We take the exponent of the means to get a better comparison of the relative values of the coefficients. We then compare this with the intuitive meaning that was described earlier. 

````julia
# Taking the first chain
chain = chains[:,:,1]

# Calculating the exponentiated means
b0_exp = exp(mean(chain[:b0]))
````


````
Error: MethodError: no method matching iterate(::Chains{Union{Missing, Floa
t64},Missing,NamedTuple{(:parameters,),Tuple{Array{String,1}}},NamedTuple{(
:samples, :hashedsummary),Tuple{Array{Turing.Utilities.Sample,1},Base.RefVa
lue{Tuple{UInt64,ChainDataFrame}}}}})
Closest candidates are:
  iterate(!Matched::Core.SimpleVector) at essentials.jl:568
  iterate(!Matched::Core.SimpleVector, !Matched::Any) at essentials.jl:568
  iterate(!Matched::ExponentialBackOff) at error.jl:199
  ...
````



````julia
b1_exp = exp(mean(chain[:b1]))
````


````
Error: MethodError: no method matching iterate(::Chains{Union{Missing, Floa
t64},Missing,NamedTuple{(:parameters,),Tuple{Array{String,1}}},NamedTuple{(
:samples, :hashedsummary),Tuple{Array{Turing.Utilities.Sample,1},Base.RefVa
lue{Tuple{UInt64,ChainDataFrame}}}}})
Closest candidates are:
  iterate(!Matched::Core.SimpleVector) at essentials.jl:568
  iterate(!Matched::Core.SimpleVector, !Matched::Any) at essentials.jl:568
  iterate(!Matched::ExponentialBackOff) at error.jl:199
  ...
````



````julia
b2_exp = exp(mean(chain[:b2]))
````


````
Error: MethodError: no method matching iterate(::Chains{Union{Missing, Floa
t64},Missing,NamedTuple{(:parameters,),Tuple{Array{String,1}}},NamedTuple{(
:samples, :hashedsummary),Tuple{Array{Turing.Utilities.Sample,1},Base.RefVa
lue{Tuple{UInt64,ChainDataFrame}}}}})
Closest candidates are:
  iterate(!Matched::Core.SimpleVector) at essentials.jl:568
  iterate(!Matched::Core.SimpleVector, !Matched::Any) at essentials.jl:568
  iterate(!Matched::ExponentialBackOff) at error.jl:199
  ...
````



````julia
b3_exp = exp(mean(chain[:b3]))
````


````
Error: MethodError: no method matching iterate(::Chains{Union{Missing, Floa
t64},Missing,NamedTuple{(:parameters,),Tuple{Array{String,1}}},NamedTuple{(
:samples, :hashedsummary),Tuple{Array{Turing.Utilities.Sample,1},Base.RefVa
lue{Tuple{UInt64,ChainDataFrame}}}}})
Closest candidates are:
  iterate(!Matched::Core.SimpleVector) at essentials.jl:568
  iterate(!Matched::Core.SimpleVector, !Matched::Any) at essentials.jl:568
  iterate(!Matched::ExponentialBackOff) at error.jl:199
  ...
````



````julia

print("The exponent of the meaned values of the weights (or coefficients are): \n")
````


````
The exponent of the meaned values of the weights (or coefficients are):
````



````julia
print("b0: ", b0_exp, " \n", "b1: ", b1_exp, " \n", "b2: ", b2_exp, " \n", "b3: ", b3_exp, " \n")
````


````
Error: UndefVarError: b0_exp not defined
````



````julia
print("The posterior distributions obtained after sampling can be visualised as :\n")
````


````
The posterior distributions obtained after sampling can be visualised as :
````




 Visualising the posterior by plotting it:

````julia
plot(chains)
````


![](/tutorials/figures/7_PoissonRegression_10_1.png)


# Interpreting the Obtained Mean Values
The exponentiated mean of the coefficient `b1` is roughly half of that of `b2`. This makes sense because in the data that we generated, the number of sneezes was more sensitive to the medicinal intake as compared to the alcohol consumption. We also get a weaker dependence on the interaction between the alcohol consumption and the medicinal intake as can be seen from the value of `b3`.


# Removing the Warmup Samples

As can be seen from the plots above, the parameters converge to their final distributions after a few iterations. These initial values during the warmup phase increase the standard deviations of the parameters and are not required after we get the desired distributions. Thus, we remove these warmup values and once again view the diagnostics. 

To remove these warmup values, we take all values except the first 200. This is because we set the second parameter of the NUTS sampler (which is the number of adaptations) to be equal to 200. `describe(chains)` is used to view the standard deviations in the estimates of the parameters. It also gives other useful information such as the means and the quantiles.

````julia
#Note the standard deviation before removing the warmup samples
describe(chains)
````


````
2-element Array{ChainDataFrame,1}

Summary Statistics
. Omitted printing of 1 columns
│ Row │ parameters │ mean      │ std        │ naive_se    │ mcse       │
│     │ Symbol     │ Float64   │ Float64    │ Float64     │ Float64    │
├─────┼────────────┼───────────┼────────────┼─────────────┼────────────┤
│ 1   │ b0         │ 1.65616   │ 0.100185   │ 0.00100185  │ 0.00503861 │
│ 2   │ b1         │ 0.554498  │ 0.124933   │ 0.00124933  │ 0.00704223 │
│ 3   │ b2         │ 0.890379  │ 0.099313   │ 0.00099313  │ 0.00567049 │
│ 4   │ b3         │ 0.266706  │ 0.0959565  │ 0.000959565 │ 0.00550219 │
│ 5   │ lf_eps     │ 0.0117511 │ 0.00881986 │ 8.81986e-5  │ 0.00081771 │

Quantiles
. Omitted printing of 1 columns
│ Row │ parameters │ 2.5%       │ 25.0%     │ 50.0%      │ 75.0%     │
│     │ Symbol     │ Float64    │ Float64   │ Float64    │ Float64   │
├─────┼────────────┼────────────┼───────────┼────────────┼───────────┤
│ 1   │ b0         │ 1.60276    │ 1.64499   │ 1.66497    │ 1.68465   │
│ 2   │ b1         │ 0.433299   │ 0.513197  │ 0.546945   │ 0.582117  │
│ 3   │ b2         │ 0.779827   │ 0.850935  │ 0.883772   │ 0.917298  │
│ 4   │ b3         │ 0.168101   │ 0.234231  │ 0.266756   │ 0.300742  │
│ 5   │ lf_eps     │ 0.00303898 │ 0.0035558 │ 0.00885919 │ 0.0196162 │
````



````julia
# Removing the first 200 values of the chains
chains_new = chains[201:2500,:,:]
describe(chains_new)
````


````
2-element Array{ChainDataFrame,1}

Summary Statistics
. Omitted printing of 1 columns
│ Row │ parameters │ mean      │ std        │ naive_se    │ mcse        │
│     │ Symbol     │ Float64   │ Float64    │ Float64     │ Float64     │
├─────┼────────────┼───────────┼────────────┼─────────────┼─────────────┤
│ 1   │ b0         │ 1.66534   │ 0.0283822  │ 0.000295904 │ 0.00112915  │
│ 2   │ b1         │ 0.545759  │ 0.0518339  │ 0.000540406 │ 0.00293611  │
│ 3   │ b2         │ 0.882892  │ 0.0491197  │ 0.000512108 │ 0.0027571   │
│ 4   │ b3         │ 0.268301  │ 0.0479853  │ 0.000500281 │ 0.00279344  │
│ 5   │ lf_eps     │ 0.0117602 │ 0.00847608 │ 8.83693e-5  │ 0.000888486 │

Quantiles

│ Row │ parameters │ 2.5%       │ 25.0%     │ 50.0%    │ 75.0%     │ 97.5% 
    │
│     │ Symbol     │ Float64    │ Float64   │ Float64  │ Float64   │ Float6
4   │
├─────┼────────────┼────────────┼───────────┼──────────┼───────────┼───────
────┤
│ 1   │ b0         │ 1.61132    │ 1.64579   │ 1.66522  │ 1.68465   │ 1.7215
9   │
│ 2   │ b1         │ 0.439848   │ 0.513478  │ 0.546904 │ 0.581287  │ 0.6420
88  │
│ 3   │ b2         │ 0.780957   │ 0.851067  │ 0.883823 │ 0.916287  │ 0.9745
39  │
│ 4   │ b3         │ 0.178882   │ 0.234922  │ 0.266776 │ 0.300027  │ 0.3636
07  │
│ 5   │ lf_eps     │ 0.00303898 │ 0.0034266 │ 0.011586 │ 0.0199196 │ 0.0208
297 │
````




Visualising the new posterior by plotting it:

````julia
plot(chains_new)
````


![](/tutorials/figures/7_PoissonRegression_13_1.png)


As can be seen from the numeric values and the plots above, the standard deviation values have decreased and all the plotted values are from the estimated posteriors. The exponentiated mean values, with the warmup samples removed, have not changed by much and they are still in accordance with their intuitive meanings as described earlier.
