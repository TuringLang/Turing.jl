## a lazy, infinite collection of iid params
type IArray
  distr :: Distribution
  vals  :: Dict{Int,Any}
end

DD = Union{Distributions.Distribution{Distributions.Univariate, Distributions.ValueSupport},
           Distributions.Distribution{Distributions.Multivariate, Distributions.ValueSupport},
           Distributions.Distribution{Distributions.Matrixvariate, Distributions.ValueSupport},
           ConjugatePriors.NormalGamma, ConjugatePriors.NormalWishart,
           ConjugatePriors.NormalInverseGamma, ConjugatePriors.NormalInverseWishart}

IArray(distr::Distribution) = IArray(distr, Dict{Int,Any}())
IArray(distr::Distribution, vals::Dict{Int,Any}) = IArray(distr, vals)
Distributions.logpdf(x :: IArray, t :: Bool) = mapreduce(v -> logpdf(x.distr, v, t), +, values(x.vals))
Distributions.logpdf(d :: DD, x :: IArray, t :: Bool) = logpdf(x :: IArray, t :: Bool)


function Base.getindex(x :: IArray, i)
  global sampler
  if i in keys(x.vals)
    x.vals[i]
  else
    x.vals[i] = rand(current_trace(), x.distr)
  end
end

# This function is not compitable with replaying.
# Base.setindex!(x :: IArray, val, key) = ( x.vals[key] = val )
