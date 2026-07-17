"""
    TracedRNG{R,N,T}

Wrapped random number generator from Random123 to keep track of random streams during model
evaluation
"""
mutable struct TracedRNG{R,N,T<:Random123.AbstractR123} <: Random.AbstractRNG
    count::Int
    rng::T
    keys::Array{R,N}
end

"""
    TracedRNG(rng::AbstractR123)

Given a counter based RNG `AbstractR123`, construct a `TracedRNG` 
"""
function TracedRNG(rng::Random123.AbstractR123{T}) where {T<:Unsigned}
    Random123.set_counter!(rng, 0)
    return TracedRNG(1, rng, T[])
end

"""
    TracedRNG([rng::AbstractRNG])

Create a `TracedRNG` with `rng` from a provided `AbstractRNG` which generates a seed used to
populate the inner counter-based RNG
"""
function TracedRNG(rng::AbstractRNG=Random.default_rng())
    inner_rng = Random.seed!(Random123.Philox2x(), rand(rng, Random.Sampler(rng, UInt64)))
    return TracedRNG(inner_rng)
end

# Connect to the Random API
Random.rng_native_52(trng::TracedRNG) = Random.rng_native_52(trng.rng)
@inline function Random.rand(trng::TracedRNG, ::Type{T}) where {T<:Unsigned}
    return Random.rand(trng.rng, T)
end

# split `key` into `n` new keys
split(key::Integer, n::Integer=1) = rand(Random.MersenneTwister(key), typeof(key), n)

# load state from current model iteration. Random streams are now replayed
load_state!(trng::TracedRNG) = Random.seed!(trng, trng.keys[trng.count])

# re-seeding the trace implies resetting the counter, see load_state! for uses
function Random.seed!(trng::TracedRNG, key)
    Random.seed!(trng.rng, key)
    return Random123.set_counter!(trng.rng, 0)
end

# add current key of the inner rng in `r` to `keys`
save_state!(trng::TracedRNG) = push!(trng.keys, state(trng.rng))

state(rng::Random123.Philox2x) = rng.key

# set the counter of the inner rng in `r`, used to keep track of the current model step
Random123.set_counter!(trng::TracedRNG, n::Integer) = trng.count = n

# increase the model step counter by `n`
inc_counter!(trng::TracedRNG, n::Integer=1) = trng.count += n
