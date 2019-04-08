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


mvnormlogpdf(u::Array, M::Array, x::Array) = logpdf(MvNormal(u, PDMat(M)), x)
mvnormlogpdf(u::Tracker.TrackedArray, M::Tracker.TrackedArray, x::Tracker.TrackedArray) =
    Tracker.track(mvnormlogpdf, u, M, x)

Tracker.@grad function mvnormlogpdf(
    u::Tracker.TrackedArray,
    M::Tracker.TrackedArray,
    x::Tracker.TrackedArray
)
    return mvnormlogpdf(Tracker.data(u), Tracker.data(M), Tracker.data(x)), function(Δ)
        inv_M = inv(M)
        z = u - x
        du = - inv_M * z
        dM = (inv_M * z * z' * inv_M - inv_M)/2
        (du, dM, -du) .* Δ
    end
end
