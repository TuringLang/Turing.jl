using Distributions
import Distributions: logpdf, pdf, sample

using ConjugatePriors
import ConjugatePriors: NormalInverseGamma, NormalWishart, NormalInverseWishart, posterior

include("transform.jl")

import Base.LinAlg: Cholesky
import Base.Random: rand
import Base: getindex, mean, scale

export NormalInverseGamma, NormalWishart, NormalInverseWishart, posterior

flatten(x :: Float64, d :: Distribution) = ( [link(d,x)], a -> invlink(d,a[1]) )
flatten(xs :: Array{Float64}, d :: Distribution) = ( link(d,xs), a -> invlink(d,a) )

function flatten(xs :: Matrix{Float64}, d :: Distribution)
  ys = link(d,xs)
  shape = size(ys)
  rebuild(zs) = invlink(d, reshape(zs, shape))
  return (ys[:], rebuild)
end

function flatten(x :: Tuple{Real,Real}, d :: NormalInverseGamma)
  y = link(d,x)
  z = [y[1],y[2]]
  return (z, a -> invlink(d, (a[1],a[2])))
end
