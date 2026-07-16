const _BASE_RNG = Random123.Philox2x

"""
    TracedRNG{R,N,T}

Wrapped random number generator from Random123 to keep track of random streams during model
evaluation
"""
mutable struct TracedRNG{R,N,T<:Random123.AbstractR123} <: Random.AbstractRNG
    count::Int
    rng::T
    keys::Array{R,N}
    refseed::Union{R,Nothing}
end

function TracedRNG(rng::Random123.AbstractR123{T}) where {T<:Unsigned}
    Random123.set_counter!(rng, 0)
    return TracedRNG(1, rng, T[], nothing)
end

"""
    TracedRNG([rng::AbstractRNG])
Create a `TracedRNG` with `rng` from a provided `AbstractRNG` which generates a seed used to
populate the inner RNG. Alternatively, one can provide a counter based RNG `AbstractR123`
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

"""
    split(key::Integer, n::Integer=1)

Split `key` into `n` new keys
"""
function split(key::Integer, n::Integer=1)
    T = typeof(key)
    inner_rng = Random.MersenneTwister(key)
    return rand(inner_rng, T, n)
end

"""
    load_state!(r::TracedRNG)

Load state from current model iteration. Random streams are now replayed
"""
function load_state!(trng::TracedRNG)
    key = trng.keys[trng.count]
    return Random.seed!(trng, key)
end

"""
    Random.seed!(rng::TracedRNG, key)

Set key and counter of inner rng in `rng` to `key` and the running model step to 0
"""
function Random.seed!(trng::TracedRNG, key)
    Random.seed!(trng.rng, key)
    return Random123.set_counter!(trng.rng, 0)
end

"""
    save_state!(r::TracedRNG)

Add current key of the inner rng in `r` to `keys`.
"""
function save_state!(trng::TracedRNG)
    return push!(trng.keys, state(trng.rng))
end

state(rng::Random123.Philox2x) = rng.key
state(rng::Random123.Philox4x) = (rng.key1, rng.key2)

function Base.copy(trng::TracedRNG)
    return TracedRNG(trng.count, copy(trng.rng), deepcopy(trng.keys), trng.refseed)
end

# Add an extra seed to the reference particle keys array to use as an alternative stream
# (we don't need to tack this one)

# We have to be careful when spliting the reference particle. Since we don't know the seed
# tree from the previous SMC run we cannot reuse any of the intermediate seed in the
# TracedRNG container. We might collide with a previous seed and the children particle would
# collapse to the reference particle. A solution to solve this is to have an extra stream
# attached to the reference particle that we only use to seed the children of the reference
# particle.

safe_set_refseed!(trng::TracedRNG{R}, seed::R) where {R} = trng.refseed = seed
safe_get_refseed(trng::TracedRNG) = trng.refseed

"""
    set_counter!(r::TracedRNG, n::Integer)

Set the counter of the inner rng in `r`, used to keep track of the current model step
"""
Random123.set_counter!(trng::TracedRNG, n::Integer) = trng.count = n

"""
    inc_counter!(r::TracedRNG, n::Integer=1)

Increase the model step counter by `n`
"""
inc_counter!(trng::TracedRNG, n::Integer=1) = trng.count += n
