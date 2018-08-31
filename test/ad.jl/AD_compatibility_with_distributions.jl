using ForwardDiff
using Distributions

# Real

x_real = randn(5)

dists = [Normal(0, 1)]

for dist in dists

    f(x) = begin
      lp = 0.0
      for i = 1:length(x)
        lp += logpdf(dist, x[i])
      end
      lp
    end

    ForwardDiff.gradient(f, x_real)

end

# Postive

x_positive = randn(5).^2

dists = [Gamma(2, 3)]

for dist in dists

    f(x) = sum(logpdf.(dist, x))

    g = x -> ForwardDiff.gradient(f, x)

end
