using Distributions, Plots, Colors

# Gaussian with known variance
# mu ~ Normal(mu0, sig0)
# x_i ~ Normal(mu, sig_like)
# Ref: https://en.wikipedia.org/wiki/Conjugate_prior#Continuous_distributions
function compute_posterior_parameters(xs, mu0, sig0, sig_like)
    n = length(xs)
    var = inv(1 / sig0^2 + n / sig_like^2)
    mu = var * (mu0 / sig0^2 + sum(xs) / sig_like^2)
    return mu, sqrt(var)
end

mu0 = 0.0
sig0 = 1.0
sig_like = 1.0
prior = Normal(mu0, sig0)

# Inference
obs = [5.0]
posterior_1 = Normal(compute_posterior_parameters(obs, mu0, sig0, sig_like)...)

# Inference with more data
push!(obs, 1.0)
posterior_2 = Normal(compute_posterior_parameters(obs, mu0, sig0, sig_like)...)

julia_purple = parse(Colorant, RGBA(0.702, 0.322, 0.8))
julia_brown = parse(Colorant, RGBA(0.8, 0.2, 0.2))
julia_green = parse(Colorant, RGBA(0.133, 0.541, 0.133))

xs = -3:0.01:5
lw = 15.0
plot(xs, pdf.(Ref(prior), xs), color=julia_purple, linewidth=lw, lab="prior", grid=false, legend=false, background_color = RGBA(1, 1, 1, 0))
plot!(xs, pdf.(Ref(posterior_1), xs), color=julia_brown, linewidth=lw, lab="posterior 1")
plot!(xs, pdf.(Ref(posterior_2), xs), color=julia_green, linewidth=lw, lab="posterior 2")
xaxis!(false)
yaxis!(false)

savefig("turing-logo.svg")
