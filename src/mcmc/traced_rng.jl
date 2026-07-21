#
# A counter-based RNG that records the seed used at each model step, so that a particle's
# trajectory can be replayed exactly. This is what lets a conditional SMC sweep reproduce
# its reference trajectory: the reference simply replays its recorded seeds.
#

"""
    TracedRNG([rng = Random.default_rng()])

A `Random123.Philox2x` generator that remembers the seed (`key`) it used at each model step
in `keys`, indexed by the step counter `count`.

  - [`save_state!`](@ref) records the current seed (ordinary particles);
  - [`load_state!`](@ref) restores `keys[count]`, replaying that step's randomness (the
    reference trajectory).
"""
mutable struct TracedRNG{K,T<:Random123.AbstractR123} <: Random.AbstractRNG
    count::Int
    rng::T
    keys::Vector{K}
end

function TracedRNG(inner::Random123.AbstractR123{T}) where {T<:Unsigned}
    Random123.set_counter!(inner, 0)
    return TracedRNG(1, inner, T[])
end
function TracedRNG(rng::AbstractRNG=Random.default_rng())
    inner = Random.seed!(Random123.Philox2x(), rand(rng, Random.Sampler(rng, UInt64)))
    return TracedRNG(inner)
end

Random.rng_native_52(trng::TracedRNG) = Random.rng_native_52(trng.rng)
Random.rand(trng::TracedRNG, ::Type{T}) where {T<:Unsigned} = Random.rand(trng.rng, T)

"The current seed of the inner generator."
inner_key(rng::Random123.Philox2x) = rng.key

"Reseed and rewind the inner generator. The model-step counter is left untouched."
function Random.seed!(trng::TracedRNG, key)
    Random.seed!(trng.rng, key)
    Random123.set_counter!(trng.rng, 0)
    return trng
end

"Record the seed used at the current step."
save_state!(trng::TracedRNG) = push!(trng.keys, inner_key(trng.rng))

"Replay the seed recorded at the current step."
load_state!(trng::TracedRNG) = Random.seed!(trng, trng.keys[trng.count])

"Set / advance the model-step counter."
set_step!(trng::TracedRNG, n::Integer) = (trng.count = n; trng)
inc_step!(trng::TracedRNG, n::Integer=1) = (trng.count += n; trng)

"Deterministically derive a fresh seed from `key`."
split_key(key::Integer) = rand(Random.MersenneTwister(key), typeof(key))

"Reseed from the generator's own current state (used between steps when not resampling)."
refresh!(trng::TracedRNG) = Random.seed!(trng, split_key(inner_key(trng.rng)))
