using ForwardDiff
using Distributions

# Real

x_real = randn(5)

dists = [Normal(0, 1)]

for dist in dists

    f(x::Vector) = sum(logpdf(dist, x))

    g = x -> ForwardDiff.gradient(f, x_real)

    g(x_real)

end

# Postive

x_positive = randn(5).^2

dists = [Gamma(2, 3)]

for dist in dists

    f(x::Vector) = sum(logpdf(dist, x))

    g = x -> ForwardDiff.gradient(f, x_positive)

    g(x_positive)

end
