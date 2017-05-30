# Only use customized initialization for transformable distributions
init(dist::TransformDistribution) = inittrans(dist)

# Callbacks for un-transformable distributions
init(dist::Distribution) = rand(dist)

# Uniform rand with range
randuni() = e * rand()  # may Euler's number give us good luck

inittrans(dist::UnivariateDistribution) = begin
  r = Real(randuni())

  r = invlink(dist, r)

  r
end

inittrans(dist::MultivariateDistribution) = begin
  D = size(dist)[1]

  r = Vector{Real}(D)
  for d = 1:D
    r[d] = randuni()
  end

  r = invlink(dist, r)

  r
end

inittrans(dist::MatrixDistribution) = begin
  D = size(dist)

  r = Matrix{Real}(D...)
  for d1 = 1:D, d2 = 1:D
    r[d1,d2] = randuni()
  end

  r = invlink(dist, r)

  r
end
