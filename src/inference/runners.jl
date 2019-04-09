function _getdist(dists::Vector{<:Distribution})
    @assert length(dists) == 1 "[observe] Turing only support vectorizing iid distribution."
    return first(dists)
end

##
# Default definition when runner = nothing
##

function observe(::Nothing, dist::Distribution, value, vi::VarInfo)
    vi.num_produce += one(vi.num_produce)
    return logpdf(dist, value)
end

function observe(::Nothing, dists::Vector{<:UnivariateDistribution}, values, vi::VarInfo)
    dist = _getdist(dists)
    return sum(logpdf.(dist, values))
end

# NOTE: this is necessary as we cannot use broadcasting for MV dists.
function observe(::Nothing, dists::Vector{<:MultivariateDistribution}, values, vi::VarInfo)
    dist = _getdist(dists)
    return sum(logpdf(dist, values))
end

##
# Sample from prior
##

function assume(spl::SampleFromPrior, dist::Distribution, vn::VarName, vi::VarInfo)

    if haskey(vi, vn)
        r = vi[vn]
    else
        r = rand(dist)
        push!(vi, vn, r, dist)
    end

    return r, logpdf_with_trans(dist, r, istrans(vi, vn))
end

function assume(spl::SampleFromUniform, dist::Distribution, vn::VarName, vi::VarInfo)

    if haskey(vi, vn)
        r = vi[vn]
    else
        r = init(dist)
        push!(vi, vn, r, dist)
    end
    # NOTE: The importance weight is not correctly computed here because
    #       r is genereated from some uniform distribution which is different from the prior

    return r, logpdf_with_trans(dist, r, istrans(vi, vn))
end

function assume(spl::A,
    dists::Vector{T},
    vn::VarName,
    var::Any,
    vi::VarInfo) where {T<:Distribution, A<:Union{SampleFromPrior, SampleFromUniform}}

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
                push!(vi, vns[i], rs[i], dist)
            end
            @assert size(var) == size(rs) "Turing.assume: variable and random number dimension unmatched"
            var = rs
        elseif isa(dist, MultivariateDistribution)
            for i = 1:n
                push!(vi, vns[i], rs[:,i], dist)
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

    var, sum(logpdf_with_trans(dist, rs, istrans(vi, vns[1])))

end

@inline observe(spl::SampleFromPrior, dist, value, vi) = observe(nothing, dist, value, vi)
@inline observe(spl::SampleFromUniform, dist, value, vi) = observe(nothing, dist, value, vi)

###
# Functions for runner to compute the log joint.
###

function assume(spl::ComputeLogJointDensity, dist::Distribution, vn::VarName, vi::VarInfo)
    @assert haskey(vi, vn)
    r = vi[vn]
    return r, logpdf_with_trans(dist, r, istrans(vi, vn))
end

function assume(spl::ComputeLogJointDensity,
    dists::AbstractVector{<:Distribution},
    vn::VarName,
    var,
    vi::VarInfo)

    @assert length(dists) == 1 "[assume] Turing only supports vectorizing iid distributions"

    dist = first(dist)
    N = size(var)[end]

    vns = map(i -> copybyindex(vn, "[$i]"), 1:N)
    rs = vi[vns]

    if isa(dist, UnivariateDistribution) || isa(dist, MatrixDistribution)
        @assert size(var) == size(rs) "[assume] Variable and random number dimension unmatched."
        var = rs
    elseif isa(dist, MultivariateDistribution)
        if isa(var, Vector)
            @assert length(var) == size(rs)[2] "[assume] Variable and random number dimension unmatched."
            for i = 1:N
                @inbounds var[i] = rs[:,i]
            end
        elseif isa(var, Matrix)
            @assert size(var) == size(rs) "[assume] Variable and random number dimension unmatched."
            var = rs
        else
            @error "[assume] Unsupported variable container."
        end
    else
        @error "[assume] Unsupported distribution type."
    end

    return var, sum(logpdf_with_trans(dist, rs, istrans(vi, first(vns))))
end

@inline observe(spl::ComputeLogJointDensity, dist, value, vi) = observe(nothing, dist, value, vi)
