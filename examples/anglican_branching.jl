## Branching example from Anglican
# https://github.com/probprog/anglican-examples/blob/master/worksheets/branching_aistats.clj

using Turing
using Distributions
using Distances

function normalize!(x)
  norm = sum(x)
  x /= norm
  return x
end

function align(x,y)
  if length(x) < length(y)
    z = zeros(y)
    z[1:length(x)] = x
    x = z
  elseif length(x) > length(y)
    z = zeros(x)
    z[1:length(y)] = y
    y = z
  end

  return (x,y)
end

function kl(p :: Categorical, q :: Categorical)
  a,b = align(p.p, q.p)
  return kl_divergence(a,b)
end

function fib(n)
  if n < 3
    return 1
  else
    fs = Array{Int64}(n)
    fs[1] = 1
    fs[2] = 2
    for i = 3:n
      fs[i] = fs[i-1] + fs[i-2]
    end
    return fs[n]
  end
end

@model anglican_branching begin
  count_prior = Poisson(4)
  @assume r ~ count_prior
  l = 0
  if 4 < r
    l = 6
  else
    @assume t ~ count_prior
    l = fib(3 * r) + t
  end
  @observe 6 ~ Poisson(l)
  @predict l
end

# Exact posterior on l (for range 0-15) copied from Anglican repo
# Seems to be wrong
anglican_branching_exact =
  Categorical(normalize!(map(x -> exp(x), [-3.9095, -2.1104, -2.6806, -Inf, -Inf, -1.1045,
                                             -1.5051, -2.0530, -2.7665, -3.5635, -4.4786,
                                             -5.5249, -6.5592, -7.8998, -8.7471])))

function anglican_branching_evaluate(results)
  weights = map(x -> x.weight, results.value)
  samples = map(x -> x.value[:l], results.value)

  fitted = fit_mle(Categorical, samples, weights)

  KL = kl(fitted, anglican_branching_exact)

  summary = Dict{Symbol,Any}()
  summary[:exact] = anglican_branching_exact
  summary[:fitted] = fitted
  summary[:KL] = KL
  return summary
end
