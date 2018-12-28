# Uniform rand with range e
randrealuni() = Real(MathConstants.e * rand())  # may Euler's number give us good luck
randrealuni(args...) = map(Real, MathConstants.e * rand(args...))

const Transformable = Union{TransformDistribution, SimplexDistribution, PDMatDistribution}


#################################
# Single-sample initialisations #
#################################

init(dist::Transformable) = inittrans(dist)
init(dist::Distribution) = rand(dist)

inittrans(dist::UnivariateDistribution) = invlink(dist, randrealuni())
inittrans(dist::MultivariateDistribution) = invlink(dist, randrealuni(size(dist)[1]))
inittrans(dist::MatrixDistribution) = invlink(dist, randrealuni(size(dist)...))


################################
# Multi-sample initialisations #
################################

init(dist::Transformable, n::Int) = inittrans(dist, n)
init(dist::Distribution, n::Int) = rand(dist, n)

inittrans(dist::UnivariateDistribution, n::Int) = invlink(dist, randrealuni(n))
function inittrans(dist::MultivariateDistribution, n::Int)
    return invlink(dist, randrealuni(size(dist)[1], n))
end
function inittrans(dist::MatrixDistribution, n::Int)
    return invlink(dist, [randrealuni(size(dist)...) for _ in 1:n])
end
