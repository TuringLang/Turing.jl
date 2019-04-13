function _getdist(dists::Vector{<:Distribution})
    @assert length(dists) == 1 "[_getdist] Turing only support vectorizing iid distribution."
    return first(dists)
end

####################
# runner = nothing #
####################

function observe(::Nothing, dist::Distribution, value, vi::VarInfo)
    vi.num_produce += one(vi.num_produce)
    return logpdf(dist, value)
end

function observe(::Nothing, dists::Vector{<:UnivariateDistribution}, values, vi::VarInfo)
    # TODO: We should probably increase num_produce here.
    dist = _getdist(dists)
    return sum(logpdf.(dist, values))
end

# NOTE: this is necessary as we cannot use broadcasting for MV dists.
function observe(::Nothing, dists::Vector{<:MultivariateDistribution}, values, vi::VarInfo)
    # TODO: We should probably increase num_produce here.
    dist = _getdist(dists)
    return sum(logpdf(dist, values))
end

###############################
# Sample from prior / uniform #
###############################

function assume(spl::SampleFromDistribution, dist::Distribution, vn::VarName, vi::VarInfo)

    if !haskey(vi, vn)
        push!(vi, vn, _rand(dist), dist)
    end

    r = vi[vn]
    return r, logpdf_with_trans(dist, r, istrans(vi, vn))
end

function assume(spl::SampleFromDistribution,
                dists::Vector{<:Union{UnivariateDistribution, MatrixDistribution}},
                vn::VarName,
                var,
                vi::VarInfo)

    dist = _getdist(dists)
    n = size(var)[end]

    vns = map(i -> copybyindex(vn, "[$i]"), 1:n)

    if haskey(vi, first(vns))
        rs = vi[vns]
    else
        rs = _rand(dist, n)
        @assert size(var) == size(rs) "[assume]: Variable and random number dimension unmatched"

        for i = 1:n
            @inbounds push!(vi, vns[i], rs[i], dist)
        end
    end

    if dist isa UnivariateDistribution
        var = rs
    else
        @inbounds var[:] .= rs[:]
    end

    return var, sum(logpdf_with_trans(dist, rs, istrans(vi, vns[1])))
end

function assume(spl::SampleFromDistribution,
                dists::Vector{<:MultivariateDistribution},
                vn::VarName,
                var::AbstractArray,
                vi::VarInfo)

    dist = _getdist(dists)
    n = size(var)[end]

    vns = map(i -> copybyindex(vn, "[$i]"), 1:n)

    if haskey(vi, first(vns))
        rs = vi[vns]
    else
        rs = _rand(dist, n)
        @assert size(var) == size(rs) "[assume]: Variable and random number dimension unmatched"

        @inbounds begin
            for i = 1:n
                push!(vi, vns[i], rs[:,i], dist)
            end

            if var isa Vector
                @assert length(var) == size(rs)[2] "Turing.assume: variable and random number dimension unmatched"
                for i = 1:n
                    var[i] = rs[:,i]
                end
            else
                @assert size(var) == size(rs) "Turing.assume: variable and random number dimension unmatched"
                var[:] .= rs[:]
            end
        end
    end

    return var, sum(logpdf_with_trans(dist, rs, istrans(vi, vns[1])))
end

@inline observe(spl::SampleFromDistribution, dist, value, vi) = observe(nothing, dist, value, vi)

#################################
# Compute the log joint Runner. #
#################################

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

    dist = _getdist(dists)
    n = size(var)[end]

    vns = map(i -> copybyindex(vn, "[$i]"), 1:n)
    @assert all(haskey.(vi, vn))

    @inbounds begin
        rs = vi[vns]

        if (dist isa MultivariateDistribution) && (var isa AbstractVector)
            @assert length(var) == last(size(rs))
            for i in 1:n
                inbounds var[i][:] .= rs[:,i]
            end
        elseif dist isa UnivariateDistribution
            var = rs
        else
            @assert size(var) == size(rs) "[assume] Variable and random number dimension unmatched."
            var[:] .= rs[:]
        end
    end

    return var, sum(logpdf_with_trans(dist, rs, istrans(vi, first(vns))))
end

@inline observe(spl::ComputeLogJointDensity, dist, value, vi) = observe(nothing, dist, value, vi)
