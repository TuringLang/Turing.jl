---
title: Bayesian Neural Networks
permalink: /:collection/:name/
---



In this tutorial, we demonstrate how one can implement a Bayesian Neural Network using a combination of Turing and [Flux](https://github.com/FluxML/Flux.jl), a suite of tools machine learning. We will use Flux to specify the neural network's layers and Turing to implement the probabalistic inference, with the goal of implementing a classification algorithm.

We will begin with importing the relevant libraries.

````julia
# Import libraries.
using Turing, Flux, Plots, Random

# Hide sampling progress.
Turing.turnprogress(false);

# Use reverse_diff due to the number of parameters in neural networks.
Turing.setadbackend(:reverse_diff)
````


````
:reverse_diff
````




Our goal here is to use a Bayesian neural network to classify points in an artificial dataset. The code below generates data points arranged in a box-like pattern and displays a graph of the dataset we'll be working with.

````julia
# Number of points to generate.
N = 80
M = round(Int, N / 4)
Random.seed!(1234)

# Generate artificial data.
x1s = rand(M) * 4.5; x2s = rand(M) * 4.5; 
xt1s = Array([[x1s[i] + 0.5; x2s[i] + 0.5] for i = 1:M])
x1s = rand(M) * 4.5; x2s = rand(M) * 4.5; 
append!(xt1s, Array([[x1s[i] - 5; x2s[i] - 5] for i = 1:M]))

x1s = rand(M) * 4.5; x2s = rand(M) * 4.5; 
xt0s = Array([[x1s[i] + 0.5; x2s[i] - 5] for i = 1:M])
x1s = rand(M) * 4.5; x2s = rand(M) * 4.5; 
append!(xt0s, Array([[x1s[i] - 5; x2s[i] + 0.5] for i = 1:M]))

# Store all the data for later.
xs = [xt1s; xt0s]
ts = [ones(2*M); zeros(2*M)]

# Plot data points.
function plot_data()
    x1 = map(e -> e[1], xt1s)
    y1 = map(e -> e[2], xt1s)
    x2 = map(e -> e[1], xt0s)
    y2 = map(e -> e[2], xt0s)

    Plots.scatter(x1,y1, color="red", clim = (0,1))
    Plots.scatter!(x2, y2, color="blue", clim = (0,1))
end

plot_data()
````


![](/tutorials/figures/3_BayesNN_2_1.png)


## Building a Neural Network

The next step is to define a [feedforward neural network](https://en.wikipedia.org/wiki/Feedforward_neural_network) where we express our parameters as distribtuions, and not single points as with traditional neural networks. The two functions below, `unpack` and `nn_forward` are helper functions we need when we specify our model in Turing.

`unpack` takes a vector of parameters and partitions them between weights and biases. `nn_forward` constructs a neural network with the variables generated in `unpack` and returns a prediction based on the weights provided.

The `unpack` and `nn_forward` functions are explicity designed to create a neural network with two hidden layers and one output layer, as shown below.


<img width="320" alt="nn-diagram" src="https://user-images.githubusercontent.com/422990/47970321-bd172080-e038-11e8-9c6d-6c2bd790bd8a.png">


The end of this tutorial provides some code that can be used to generate more general network shapes.

````julia
# Turn a vector into a set of weights and biases.
function unpack(nn_params::AbstractVector)
    W₁ = reshape(nn_params[1:6], 3, 2);   
    b₁ = reshape(nn_params[7:9], 3)
    
    W₂ = reshape(nn_params[10:15], 2, 3); 
    b₂ = reshape(nn_params[16:17], 2)
    
    Wₒ = reshape(nn_params[18:19], 1, 2); 
    bₒ = reshape(nn_params[20:20], 1)   
    return W₁, b₁, W₂, b₂, Wₒ, bₒ
end

# Construct a neural network using Flux and return a predicted value.
function nn_forward(xs, nn_params::AbstractVector)
    W₁, b₁, W₂, b₂, Wₒ, bₒ = unpack(nn_params)
    nn = Chain(Dense(W₁, b₁, tanh),
               Dense(W₂, b₂, tanh),
               Dense(Wₒ, bₒ, σ))
    return nn(xs)
end;
````




The probabalistic model specification below creates a `params` variable, which has 20 normally distributed variables. Each entry in the `params` vector represents weights and biases of our neural net.

````julia
# Create a regularization term and a Gaussain prior variance term.
alpha = 0.09
sig = sqrt(1.0 / alpha)

# Specify the probabalistic model.
@model bayes_nn(xs, ts) = begin
    # Create the weight and bias vector.
    nn_params ~ MvNormal(zeros(20), sig .* ones(20))
    
    # Calculate predictions for the inputs given the weights
    # and biases in theta.
    preds = nn_forward(xs, nn_params)
    
    # Observe each prediction.
    for i = 1:length(ts)
        ts[i] ~ Bernoulli(preds[i])
    end
end;
````




Inference can now be performed by calling `sample`. We use the `HMC` sampler here.

````julia
# Perform inference.
N = 5000
ch = sample(bayes_nn(hcat(xs...), ts), HMC(N, 0.05, 4));
````


````
[HMC] Finished with
  Running time        = 244.2390099529995;
  Accept rate         = 0.9048;
  #lf / sample        = 3.9992;
  #evals / sample     = 5.9992;
  pre-cond. metric    = [1.0].
````




Now we extract the weights and biases from the sampled chain. We'll use these primarily in determining how good a classifier our model is.

````julia
# Extract all weight and bias parameters.
theta = ch[:nn_params].value.data;
````




## Prediction Visualization

We can use [MAP estimation](https://en.wikipedia.org/wiki/Maximum_a_posteriori_estimation) to classify our population by using the set of weights that provided the highest log posterior.

````julia
# Plot the data we have.
plot_data()

# Find the index that provided the highest log posterior in the chain.
_, i = findmax(ch[:lp].value.data)

# Extract the max row value from i.
i = i.I[1]

# Plot the posterior distribution with a contour plot.
x_range = collect(range(-6,stop=6,length=25))
y_range = collect(range(-6,stop=6,length=25))
Z = [nn_forward([x, y], theta[i, :])[1] for x=x_range, y=y_range]
contour!(x_range, y_range, Z)
````


![](/tutorials/figures/3_BayesNN_7_1.png)


The contour plot above shows that the MAP method is not too bad at classifying our data.


Now we can visualize our predictions.

\$\$ 
p(\tilde{x} | X, \alpha) = \int_{\theta} p(\tilde{x} | \theta) p(\theta | X, \alpha) \approx \sum_{\theta \sim p(\theta | X, \alpha)}f_{\theta}(\tilde{x}) 
\$\$

The `nn_predict` function takes the average predicted value from a network parameterized by weights drawn from the MCMC chain.

````julia
# Return the average predicted value across
# multiple weights.
function nn_predict(x, theta, num)
    mean([nn_forward(x, theta[i,:])[1] for i in 1:10:num])
end;
````




Next, we use the `nn_predict` function to predict the value at a sample of points where the `x` and `y` coordinates range between -6 and 6. As we can see below, we still have a satisfactory fit to our data.

````julia
# Plot the average prediction.
plot_data()

n_end = 1500
x_range = collect(range(-6,stop=6,length=25))
y_range = collect(range(-6,stop=6,length=25))
Z = [nn_predict([x, y], theta, n_end)[1] for x=x_range, y=y_range]
contour!(x_range, y_range, Z)
````


![](/tutorials/figures/3_BayesNN_9_1.png)


If you are interested in how the predictive power of our Bayesian neural network evolved between samples, the following graph displays an animation of the contour plot generated from the network weights in samples 1 to 1,000. 

````julia
# Number of iterations to plot.
n_end = 500

anim = @animate for i=1:n_end
    plot_data()
    Z = [nn_forward([x, y], theta[i,:])[1] for x=x_range, y=y_range]
    contour!(x_range, y_range, Z, title="Iteration $$i", clim = (0,1))
end every 5;
````




![3_bayesnn_anim](https://user-images.githubusercontent.com/422990/48957381-2e961080-ef0d-11e8-8c52-dbe35d812497.gif)


## Generic Bayesian Neural Networks

The below code is intended for use in more general applications, where you need to be able to change the basic network shape fluidly. The code above is highly rigid, and adapting it for other architectures would be time consuming. Currently the code below only supports networks of `Dense` layers.

Here, we solve the same problem as above, but with three additional 2x2 `tanh` hidden layers. You can modify the `network_shape` variable to specify differing architectures. A tuple `(3,2, :tanh)` means you want to construct a `Dense` layer with 3 outputs, 2 inputs, and a `tanh` activation function. You can provide any activation function found in Flux by entering it as a `Symbol` (e.g., the `tanh` function is entered in the third part of the tuple as `:tanh`).

````julia
# Specify the network architecture.
network_shape = [
    (3,2, :tanh),
    (2,3, :tanh), 
    (1,2, :σ)]

# Regularization, parameter variance, and total number of
# parameters.
alpha = 0.09
sig = sqrt(1.0 / alpha)
num_params = sum([i * o + i for (i, o, _) in network_shape])

# This modification of the unpack function generates a series of vectors
# given a network shape.
function unpack(θ::AbstractVector, network_shape::AbstractVector)
    index = 1
    weights = []
    biases = []
    for layer in network_shape
        rows, cols, _ = layer
        size = rows * cols
        last_index_w = size + index - 1
        last_index_b = last_index_w + rows
        push!(weights, reshape(θ[index:last_index_w], rows, cols))
        push!(biases, reshape(θ[last_index_w+1:last_index_b], rows))
        index = last_index_b + 1
    end
    return weights, biases
end

# Generate an abstract neural network given a shape, 
# and return a prediction.
function nn_forward(x, θ::AbstractVector, network_shape::AbstractVector)
    weights, biases = unpack(θ, network_shape)
    layers = []
    for i in eachindex(network_shape)
        push!(layers, Dense(weights[i],
            biases[i],
            eval(network_shape[i][3])))
    end
    nn = Chain(layers...)
    return nn(x)
end

# General Turing specification for a BNN model.
@model bayes_nn(xs, ts, network_shape, num_params) = begin
    θ ~ MvNormal(zeros(num_params), sig .* ones(num_params))
    preds = nn_forward(xs, θ, network_shape)
    for i = 1:length(ts)
        ts[i] ~ Bernoulli(preds[i])
    end
end

# Set the backend.
Turing.setadbackend(:reverse_diff)

# Perform inference.
num_samples = 500
ch2 = sample(bayes_nn(hcat(xs...), ts, network_shape, num_params), NUTS(num_samples, 0.65));
````


````
[NUTS] Finished with
  Running time        = 584.4946359479999;
  #lf / sample        = 0.0;
  #evals / sample     = 179.434;
  pre-cond. metric    = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0,....
````



````julia
# This function makes predictions based on network shape.
function nn_predict(x, theta, num, network_shape)
    mean([nn_forward(x, theta[i,:], network_shape)[1] for i in 1:10:num])
end;

# Extract the θ parameters from the sampled chain.
params2 = ch2[:θ].value.data

plot_data()

x_range = collect(range(-6,stop=6,length=25))
y_range = collect(range(-6,stop=6,length=25))
Z = [nn_predict([x, y], params2, num_samples, network_shape)[1] for x=x_range, y=y_range]
contour!(x_range, y_range, Z)
````


![](/tutorials/figures/3_BayesNN_12_1.png)


This has been an introduction to the applications of Turing and Flux in defining Bayesian neural networks.
