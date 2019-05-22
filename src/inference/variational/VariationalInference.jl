abstract type VariationalInference <: InferenceAlgorithm end

"""
    sample(vi::VariationalInference, num_samples)

Produces `num_samples` samples for the given VI method using number of samples equal to `num_samples`.
"""
function sample(vi::VariationalInference, num_samples) end

"""
    objective(vi::VariationalInference, num_samples)

Computes empirical estimates of ELBO for the given VI method using number of samples equal to `num_samples`.
"""
function objective(vi::VariationalInference, num_samples) end

"""
    optimize(vi::VariationalInference)

Finds parameters which maximizes the ELBO for the given VI method.
"""
function optimize(vi::VariationalInference) end

# sampler interface

function Sampler(alg::VariationalInference, s::Selector)
    info = Dict{Symbol, Any}()
    return Sampler(alg, info, s)
end

function assume(spl::Sampler{<:VariationalInference}, dist::Distribution, vn::VarName, vi::VarInfo)
    Turing.DEBUG && @debug "assuming..."
    updategid!(vi, vn, spl)
    r = vi[vn]
    # acclogp!(vi, logpdf_with_trans(dist, r, istrans(vi, vn)))
    # r
    Turing.DEBUG && @debug "dist = $dist"
    Turing.DEBUG && @debug "vn = $vn"
    Turing.DEBUG && @debug "r = $r" "typeof(r)=$(typeof(r))"

    r, logpdf_with_trans(dist, r, istrans(vi, vn))
end

observe(spl::Sampler{<:VariationalInference},
    d::Distribution,
    value::Any,
    vi::VarInfo) = observe(nothing, d, value, vi)


# concrete algorithms
include("advi.jl")
