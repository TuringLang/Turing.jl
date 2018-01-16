# Uniform rand with range e
randrealuni() = Real(e * rand())  # may Euler's number give us good luck
randrealuni(args...) = map(x -> Real(x), e * rand(args...))

# Only use customized initialization for transformable distributions
init(dist::Union{TransformDistribution,SimplexDistribution,PDMatDistribution}) = inittrans(dist)

# Callbacks for un-transformable distributions
init(dist::Distribution) = rand(dist)

inittrans(dist::UnivariateDistribution) = begin
  r = randrealuni()

  r = invlink(dist, r)

  r
end

inittrans(dist::MultivariateDistribution) = begin
  D = size(dist)[1]

  r = randrealuni(D)

  r = invlink(dist, r)

  r
end

inittrans(dist::MatrixDistribution) = begin
  D = size(dist)

  r = randrealuni(D...)

  r = invlink(dist, r)

  r
end


# Only use customized initialization for transformable distributions
init(dist::TransformDistribution, n::Int) = inittrans(dist, n)

# Callbacks for un-transformable distributions
init(dist::Distribution, n::Int) = rand(dist, n)

inittrans(dist::UnivariateDistribution, n::Int) = begin
  rs = randrealuni(n)

  rs = invlink(dist, rs)

  rs
end

inittrans(dist::MultivariateDistribution, n::Int) = begin
  D = size(dist)[1]

  rs = randrealuni(D, n)

  rs = invlink(dist, rs)

  rs
end

inittrans(dist::MatrixDistribution, n::Int) = begin
  D = size(dist)

  rs = [randrealuni(D...) for _ = 1:n]

  rs = invlink(dist, rs)

  rs
end
