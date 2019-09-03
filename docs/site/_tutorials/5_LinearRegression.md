---
title: Linear Regression
permalink: /:collection/:name/
---


Turing is powerful when applied to complex hierarchical models, but it can also be put to task at common statistical procedures, like [linear regression](https://en.wikipedia.org/wiki/Linear_regression). This tutorial covers how to implement a linear regression model in Turing.

## Set Up

We begin by importing all the necessary libraries.

````julia
# Import Turing and Distributions.
using Turing, Distributions

# Import RDatasets.
using RDatasets

# Import MCMCChains, Plots, and StatPlots for visualizations and diagnostics.
using MCMCChains, Plots, StatsPlots

# Set a seed for reproducibility.
using Random
Random.seed!(0);

# Hide the progress prompt while sampling.
Turing.turnprogress(false);
````




We will use the `mtcars` dataset from the [RDatasets](https://github.com/johnmyleswhite/RDatasets.jl) package. `mtcars` contains a variety of statistics on different car models, including their miles per gallon, number of cylinders, and horsepower, among others.

We want to know if we can construct a Bayesian linear regression model to predict the miles per gallon of a car, given the other statistics it has. Lets take a look at the data we have.

````julia
# Import the "Default" dataset.
data = RDatasets.dataset("datasets", "mtcars");

# Show the first six rows of the dataset.
first(data, 6)
````


````
6×12 DataFrame. Omitted printing of 6 columns
│ Row │ Model             │ MPG      │ Cyl    │ Disp     │ HP     │ DRat   
  │
│     │ String⍰           │ Float64⍰ │ Int64⍰ │ Float64⍰ │ Int64⍰ │ Float64
⍰ │
├─────┼───────────────────┼──────────┼────────┼──────────┼────────┼────────
──┤
│ 1   │ Mazda RX4         │ 21.0     │ 6      │ 160.0    │ 110    │ 3.9    
  │
│ 2   │ Mazda RX4 Wag     │ 21.0     │ 6      │ 160.0    │ 110    │ 3.9    
  │
│ 3   │ Datsun 710        │ 22.8     │ 4      │ 108.0    │ 93     │ 3.85   
  │
│ 4   │ Hornet 4 Drive    │ 21.4     │ 6      │ 258.0    │ 110    │ 3.08   
  │
│ 5   │ Hornet Sportabout │ 18.7     │ 8      │ 360.0    │ 175    │ 3.15   
  │
│ 6   │ Valiant           │ 18.1     │ 6      │ 225.0    │ 105    │ 2.76   
  │
````



````julia
size(data)
````


````
(32, 12)
````




The next step is to get our data ready for testing. We'll split the `mtcars` dataset into two subsets, one for training our model and one for evaluating our model. Then, we separate the labels we want to learn (`MPG`, in this case) and standardize the datasets by subtracting each column's means and dividing by the standard deviation of that column.

The resulting data is not very familiar looking, but this standardization process helps the sampler converge far easier. We also create a function called `unstandardize`, which returns the standardized values to their original form. We will use this function later on when we make predictions.

````julia
# Function to split samples.
function split_data(df, at = 0.70)
    (r, _) = size(df)
    index = Int(round(r * at))
    train = df[1:index, :]
    test  = df[(index+1):end, :]
    return train, test
end

# Split our dataset 70%/30% into training/test sets.
train, test = split_data(data, 0.7)

# Save dataframe versions of our dataset.
train_cut = DataFrame(train)
test_cut = DataFrame(test)

# Create our labels. These are the values we are trying to predict.
train_label = train[:, :MPG]
test_label = test[:, :MPG]

# Get the list of columns to keep.
remove_names = filter(x->!in(x, [:MPG, :Model]), names(data))

# Filter the test and train sets.
train = Matrix(train[:,remove_names]);
test = Matrix(test[:,remove_names]);

# A handy helper function to rescale our dataset.
function standardize(x)
    return (x .- mean(x, dims=1)) ./ std(x, dims=1), x
end

# Another helper function to unstandardize our datasets.
function unstandardize(x, orig)
    return x .* std(orig, dims=1) .+ mean(orig, dims=1)
end

# Standardize our dataset.
(train, train_orig) = standardize(train)
(test, test_orig) = standardize(test)
(train_label, train_l_orig) = standardize(train_label)
(test_label, test_l_orig) = standardize(test_label);
````




## Model Specification

In a traditional frequentist model using [OLS](https://en.wikipedia.org/wiki/Ordinary_least_squares), our model might look like:

\$\$
MPG_i = \alpha + \boldsymbol{\beta}^T\boldsymbol{X_i}
\$\$

where $$\boldsymbol{\beta}$$ is a vector of coefficients and $$\boldsymbol{X}$$ is a vector of inputs for observation $$i$$. The Bayesian model we are more concerned with is the following:

\$\$
MPG_i \sim \mathcal{N}(\alpha + \boldsymbol{\beta}^T\boldsymbol{X_i}, \sigma^2)
\$\$

where $$\alpha$$ is an intercept term common to all observations, $$\boldsymbol{\beta}$$ is a coefficient vector, $$\boldsymbol{X_i}$$ is the observed data for car $$i$$, and $$\sigma^2$$ is a common variance term.

For $$\sigma^2$$, we assign a prior of `TruncatedNormal(0,100,0,Inf)`. This is consistent with [Andrew Gelman's recommendations](http://www.stat.columbia.edu/~gelman/research/published/taumain.pdf) on noninformative priors for variance. The intercept term ($$\alpha$$) is assumed to be normally distributed with a mean of zero and a variance of three. This represents our assumptions that miles per gallon can be explained mostly by our assorted variables, but a high variance term indicates our uncertainty about that. Each coefficient is assumed to be normally distributed with a mean of zero and a variance of 10. We do not know that our coefficients are different from zero, and we don't know which ones are likely to be the most important, so the variance term is quite high. Lastly, each observation $$y_i$$ is distributed according to the calculated `mu` term given by $$\alpha + \boldsymbol{\beta}^T\boldsymbol{X_i}$$.

````julia
# Bayesian linear regression.
@model linear_regression(x, y, n_obs, n_vars) = begin
    # Set variance prior.
    σ₂ ~ TruncatedNormal(0,100, 0, Inf)
    
    # Set intercept prior.
    intercept ~ Normal(0, 3)
    
    # Set the priors on our coefficients.
    coefficients = Array{Real}(undef, n_vars)
    coefficients ~ [Normal(0, 10)]
    
    # Calculate all the mu terms.
    mu = intercept .+ x * coefficients
    for i = 1:n_obs
        y[i] ~ Normal(mu[i], σ₂)
    end
end;
````




With our model specified, we can call the sampler. We will use the No U-Turn Sampler ([NUTS](http://turing.ml/docs/library/#-turingnuts--type)) here. 

````julia
n_obs, n_vars = size(train)
model = linear_regression(train, train_label, n_obs, n_vars)
chain = sample(model, NUTS(1500, 200, 0.65));
````


````
[NUTS] Finished with
  Running time        = 28.76012935999999;
  #lf / sample        = 0.0;
  #evals / sample     = 0.0006666666666666666;
  pre-cond. metric    = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0,....
````




As a visual check to confirm that our coefficients have converged, we show the densities and trace plots for our parameters using the `plot` functionality.

````julia
plot(chain)
````


![](/tutorials/figures/5_LinearRegression_7_1.png)


It looks like each of our parameters has converged. We can check our numerical esimates using `describe(chain)`, as below.

````julia
describe(chain)
````


````
2-element Array{ChainDataFrame,1}

Summary Statistics
. Omitted printing of 1 columns
│ Row │ parameters       │ mean       │ std       │ naive_se   │ mcse      
 │
│     │ Symbol           │ Float64    │ Float64   │ Float64    │ Float64   
 │
├─────┼──────────────────┼────────────┼───────────┼────────────┼───────────
─┤
│ 1   │ coefficients[1]  │ 0.374513   │ 0.44485   │ 0.011486   │ 0.0227265 
 │
│ 2   │ coefficients[2]  │ -0.171117  │ 0.476053  │ 0.0122916  │ 0.0225355 
 │
│ 3   │ coefficients[3]  │ -0.0681829 │ 0.356122  │ 0.00919503 │ 0.0163717 
 │
│ 4   │ coefficients[4]  │ 0.66256    │ 0.33855   │ 0.00874132 │ 0.0141081 
 │
│ 5   │ coefficients[5]  │ 0.0969497  │ 0.483806  │ 0.0124918  │ 0.0278113 
 │
│ 6   │ coefficients[6]  │ 0.0400533  │ 0.272691  │ 0.00704085 │ 0.016834  
 │
│ 7   │ coefficients[7]  │ -0.0995777 │ 0.295442  │ 0.00762827 │ 0.0135998 
 │
│ 8   │ coefficients[8]  │ 0.10959    │ 0.313314  │ 0.00808972 │ 0.0171665 
 │
│ 9   │ coefficients[9]  │ 0.200219   │ 0.329276  │ 0.00850186 │ 0.0116165 
 │
│ 10  │ coefficients[10] │ -0.682739  │ 0.361389  │ 0.00933104 │ 0.0179951 
 │
│ 11  │ intercept        │ 0.0108571  │ 0.170723  │ 0.00440804 │ 0.0106107 
 │
│ 12  │ lf_eps           │ 0.0581085  │ 0.0402024 │ 0.00103802 │ 0.00132349
 │
│ 13  │ σ₂               │ 0.484513   │ 0.492571  │ 0.0127181  │ 0.036488  
 │

Quantiles
. Omitted printing of 1 columns
│ Row │ parameters       │ 2.5%      │ 25.0%       │ 50.0%      │ 75.0%    
 │
│     │ Symbol           │ Float64   │ Float64     │ Float64    │ Float64  
 │
├─────┼──────────────────┼───────────┼─────────────┼────────────┼──────────
─┤
│ 1   │ coefficients[1]  │ -0.497325 │ 0.106879    │ 0.367559   │ 0.650437 
 │
│ 2   │ coefficients[2]  │ -1.08863  │ -0.444431   │ -0.174282  │ 0.101657 
 │
│ 3   │ coefficients[3]  │ -0.808397 │ -0.294186   │ -0.0607567 │ 0.176866 
 │
│ 4   │ coefficients[4]  │ 0.028891  │ 0.453163    │ 0.669321   │ 0.847996 
 │
│ 5   │ coefficients[5]  │ -0.848829 │ -0.197623   │ 0.0904946  │ 0.384393 
 │
│ 6   │ coefficients[6]  │ -0.495648 │ -0.128853   │ 0.0474724  │ 0.200374 
 │
│ 7   │ coefficients[7]  │ -0.662909 │ -0.268329   │ -0.109192  │ 0.0712903
 │
│ 8   │ coefficients[8]  │ -0.421245 │ -0.053784   │ 0.105746   │ 0.24969  
 │
│ 9   │ coefficients[9]  │ -0.438313 │ -0.00346737 │ 0.20142    │ 0.408158 
 │
│ 10  │ coefficients[10] │ -1.38271  │ -0.88346    │ -0.679579  │ -0.460576
 │
│ 11  │ intercept        │ -0.192764 │ -0.0576108  │ 0.00142006 │ 0.0631787
 │
│ 12  │ lf_eps           │ 0.0233708 │ 0.0564162   │ 0.0564162  │ 0.0564162
 │
│ 13  │ σ₂               │ 0.293726  │ 0.369497    │ 0.435216   │ 0.508814 
 │
````




## Comparing to OLS

A satisfactory test of our model is to evaluate how well it predicts. Importantly, we want to compare our model to existing tools like OLS. The code below uses the [GLM.jl]() package to generate a traditional OLS multivariate regression on the same data as our probabalistic model.

````julia
# Import the GLM package.
using GLM

# Perform multivariate OLS.
ols = lm(@formula(MPG ~ Cyl + Disp + HP + DRat + WT + QSec + VS + AM + Gear + Carb), train_cut)

# Store our predictions in the original dataframe.
train_cut.OLSPrediction = GLM.predict(ols);
test_cut.OLSPrediction = GLM.predict(ols, test_cut);
````




The function below accepts a chain and an input matrix and calculates predictions. We use the mean observation of each parameter in the model starting with sample 200, which is where the warm-up period for the NUTS sampler ended.

````julia
# Make a prediction given an input vector.
function prediction(chain, x)
    p = get_params(chain[200:end, :, :])
    α = mean(p.intercept)
    β = collect(mean.(p.coefficients))
    return  α .+ x * β
end
````


````
prediction (generic function with 2 methods)
````




When we make predictions, we unstandardize them so they're more understandable. We also add them to the original dataframes so they can be placed in context.

````julia
# Calculate the predictions for the training and testing sets.
train_cut.BayesPredictions = unstandardize(prediction(chain, train), train_l_orig);
test_cut.BayesPredictions = unstandardize(prediction(chain, test), test_l_orig);

# Show the first side rows of the modified dataframe.
first(test_cut, 6)
````


````
6×14 DataFrame. Omitted printing of 8 columns
│ Row │ Model            │ MPG      │ Cyl    │ Disp     │ HP     │ DRat    
 │
│     │ String⍰          │ Float64⍰ │ Int64⍰ │ Float64⍰ │ Int64⍰ │ Float64⍰
 │
├─────┼──────────────────┼──────────┼────────┼──────────┼────────┼─────────
─┤
│ 1   │ AMC Javelin      │ 15.2     │ 8      │ 304.0    │ 150    │ 3.15    
 │
│ 2   │ Camaro Z28       │ 13.3     │ 8      │ 350.0    │ 245    │ 3.73    
 │
│ 3   │ Pontiac Firebird │ 19.2     │ 8      │ 400.0    │ 175    │ 3.08    
 │
│ 4   │ Fiat X1-9        │ 27.3     │ 4      │ 79.0     │ 66     │ 4.08    
 │
│ 5   │ Porsche 914-2    │ 26.0     │ 4      │ 120.3    │ 91     │ 4.43    
 │
│ 6   │ Lotus Europa     │ 30.4     │ 4      │ 95.1     │ 113    │ 3.77    
 │
````




Now let's evaluate the loss for each method, and each prediction set. We will use sum of squared error function to evaluate loss, given by 

\$\$
\text{SSE} = \sum{(y_i - \hat{y_i})^2}
\$\$

where $$y_i$$ is the actual value (true MPG) and $$\hat{y_i}$$ is the predicted value using either OLS or Bayesian linear regression. A lower SSE indicates a closer fit to the data.

````julia
bayes_loss1 = sum((train_cut.BayesPredictions - train_cut.MPG).^2)
ols_loss1 = sum((train_cut.OLSPrediction - train_cut.MPG).^2)

bayes_loss2 = sum((test_cut.BayesPredictions - test_cut.MPG).^2)
ols_loss2 = sum((test_cut.OLSPrediction - test_cut.MPG).^2)

println("Training set:
    Bayes loss: $$bayes_loss1
    OLS loss: $$ols_loss1
Test set: 
    Bayes loss: $$bayes_loss2
    OLS loss: $$ols_loss2")
````


````
Training set:
    Bayes loss: 68.00979321046889
    OLS loss: 67.56037474764624
Test set: 
    Bayes loss: 242.57948201282844
    OLS loss: 270.94813070761944
````




As we can see above, OLS and our Bayesian model fit our training set about the same. This is to be expected, given that it is our training set. But when we look at our test set, we see that the Bayesian linear regression model is better able to predict out of sample.
