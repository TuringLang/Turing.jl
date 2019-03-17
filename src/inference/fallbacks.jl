## Fallback functions

init_spl(model, alg) = Sampler(alg, nothing), VarInfo(model)

assume(spl::Sampler, dist::Distribution) =
error("Turing.assume: unmanaged inference algorithm: $(typeof(spl))")

observe(spl::Sampler, weight::Float64) =
error("Turing.Inference.observe: unmanaged inference algorithm: $(typeof(spl))")

## Default definitions for assume, observe, when sampler = nothing.
##  Note: `A<:Union{Nothing, SampleFromPrior, SampleFromUniform}` can be
##   simplified into `A<:Union{SampleFromPrior, SampleFromUniform}` when
##   all `spl=nothing` is replaced with `spl=SampleFromPrior`.
function assume(spl::A,
    dist::Distribution,
    vn::VarName,
    vi::AbstractVarInfo) where {A<:Union{Nothing, SampleFromPrior, SampleFromUniform}}

    if haskey(vi, vn)
        r = vi[vn]
    else
        r = isa(spl, SampleFromUniform) ? init(dist) : rand(dist)
        push!(vi, vn, r, dist, 0)
    end
    # NOTE: The importance weight is not correctly computed here because
    #       r is genereated from some uniform distribution which is different from the prior
    # acclogp!(vi, logpdf_with_trans(dist, r, istrans(vi, vn)))

    return r, logpdf_with_trans(dist, r, istrans(vi, vn))
end

function assume(spl::A,
    dists::Vector{T},
    vn::VarName,
    var::Any,
    vi::AbstractVarInfo) where {T<:Distribution, A<:Union{Nothing, SampleFromPrior, SampleFromUniform}}

    @assert length(dists) == 1 "Turing.assume only support vectorizing i.i.d distribution"
    dist = dists[1]
    n = size(var)[end]

    vns = map(i -> copybyindex(vn, "[$i]"), 1:n)

    if haskey(vi, vns[1])
        rs = vi[vns]
    else
        rs = isa(spl, SampleFromUniform) ? init(dist, n) : rand(dist, n)

        if isa(dist, UnivariateDistribution) || isa(dist, MatrixDistribution)
            for i = 1:n
                push!(vi, vns[i], rs[i], dist, 0)
            end
            @assert size(var) == size(rs) "Turing.assume: variable and random number dimension unmatched"
            var = rs
        elseif isa(dist, MultivariateDistribution)
            for i = 1:n
                push!(vi, vns[i], rs[:,i], dist, 0)
            end
            if isa(var, Vector)
                @assert length(var) == size(rs)[2] "Turing.assume: variable and random number dimension unmatched"
                for i = 1:n
                    var[i] = rs[:,i]
                end
            elseif isa(var, Matrix)
                @assert size(var) == size(rs) "Turing.assume: variable and random number dimension unmatched"
                var = rs
            else
                @error("Turing.assume: unsupported variable container"); error()
            end
        end
    end

    # acclogp!(vi, sum(logpdf_with_trans(dist, rs, istrans(vi, vns[1]))))

    return var, sum(logpdf_with_trans(dist, rs, istrans(vi, vns[1])))
end

function observe(spl::A,
    dist::Distribution,
    value::Any,
    vi::AbstractVarInfo) where {A<:Union{Nothing, SampleFromPrior, SampleFromUniform}}

    vi.num_produce += one(vi.num_produce)
    Turing.DEBUG && @debug "dist = $dist"
    Turing.DEBUG && @debug "value = $value"

    # acclogp!(vi, logpdf(dist, value))
    return logpdf(dist, value)
end

function observe(spl::A,
    dists::Vector{T},
    value::Any,
    vi::AbstractVarInfo) where {T<:Distribution, A<:Union{Nothing, SampleFromPrior, SampleFromUniform}}

    @assert length(dists) == 1 "Turing.Inference.observe only support vectorizing i.i.d distribution"
    dist = dists[1]
    @assert isa(dist, UnivariateDistribution) || isa(dist, MultivariateDistribution) "Turing.Inference.observe: vectorizing matrix distribution is not supported"
    if isa(dist, UnivariateDistribution)  # only univariate distributions support broadcast operation (logpdf.) by Distributions.jl
        # acclogp!(vi, sum(logpdf.(Ref(dist), value)))
        return sum(logpdf.(Ref(dist), value))
    else
        # acclogp!(vi, sum(logpdf(dist, value)))
        return sum(logpdf(dist, value))
    end
end
