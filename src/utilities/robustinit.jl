# Uniform rand with range 2; ref: https://mc-stan.org/docs/2_19/reference-manual/initialization.html
randrealuni() = Real(2rand())
randrealuni(args...) = map(Real, 2rand(args...))

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
