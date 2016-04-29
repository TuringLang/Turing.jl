"""
Function `condition` is similar to `assume`, however it does not generate new values for `vname`
from `distr`. Instead, it condition on an existing value of `vname`. This existing value can be
mutated externally by some inference method like SMC2, HMC, PMMH etc.
"""
immutable Conditional
  vals   :: Array{Distributions.VariateForm, 1}
  distrs :: Array{Distributions.Distribution,1}
  pos  :: TArray{Int64,1}  # N.B. TArray is only meaningful for ParticleSamplers, since each particle needs a counter for its passed parameters.
end

# Condition on a variable during inference, e.g. in SMC2, we sample x | theta in the inner SMC.
#  A new value is sampled from the prior, if no values for theta is found in `Sampler.conditionals`.
#  Otherwise re-use or replay the value in `Sampler.conditionals`.

# theta = condition( spl :: Sampler, distr :: Distribution )

# Utility functions

# flatten( c :: Conditional )
# logpdf (c :: Conditional ) is directly added to `Trace.logweight` in each call of `condition`

#Params does not contain a counter, since it is Xparticle specific
#the length of the arrays should be equal to the number of elements stored in them
type Params
  values::Array{Any} # values of the random variables in a run
  dists::Array{Any} # the prior on the parameters
end
Params() = Params([],[])

# copies the entire structure
## TODO: #is shallow copy enough here?
clone(ps::Params) = Params(deepcopy(ps.values), deepcopy(ps.dists))

# TODO: rename prior to logpdf
# function prior(params :: Params) #prior logdensity for the current values
#   result = 0
#   for i=1:length(params.values)
#     result += logpdf(params.dists[i], params.values[i], true)
#   end
#   return result
# end
prior(params :: Params) = reduce(+,map((d,v)->logpdf(d,v,true), params.dists, params.values))
logpdf(params :: Params) = prior(params)

#Flatten should return a flat array with all the elements and a function that can rebuild the
#original data structure from such an array.
#It takes a distribution as the second argument to compute the constraints.
#The returned values are transformed to cover the whole real line.
function flatten(ps :: Params)
  a = map(flatten, ps.values, ps.dists)
  rebuilders = map(x -> x[2], a)
  vectors = map(x -> x[1], a)
  sizes = map(length, vectors)
  flat = Float64[]
  foldl(append!, flat, vectors)
  function reb(flat)
    consumed = 0
    objects = Any[]
    while(consumed < length(flat))
      which = length(objects) + 1
      n = sizes[which]
      push!(objects, rebuilders[which](flat[(consumed + 1) : (consumed + n)]))
      consumed += n
    end
    return Params(objects, deepcopy(ps.dists))
  end
  return (flat, reb)
end
