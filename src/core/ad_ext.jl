using Tracker, Distributions, PDMats

# The code below introduces Tracker and Distributions to each other to
# make them work well together. The code can be removed after that issue
# is fixed on the Distributions side.

PDMats.PDMat(ta::TrackedArray) = PDMat([e for e in ta])
Distributions.MvNormal(ta::TrackedArray, mat::PDMat) = MvNormal([e for e in ta], mat)
Distributions.logpdf(
    dist::Distribution{Multivariate,S} where S<:ValueSupport,
    ta::TrackedArray{T,N,A} where A<:AbstractArray{T, N} where {T, N}
) = logpdf(dist, [e for e in ta])
