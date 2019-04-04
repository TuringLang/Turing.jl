###
# Functions for runner to compute the log joint.
###

function assume(spl::ComputeLogJointDensity,
    dist::Distribution,
    vn::VarName,
    vi::VarInfo)

    r = vi[vn]
    return r, logpdf_with_trans(dist, r, istrans(vi, vn))
end

function assume(spl::ComputeLogJointDensity,
    dists::AbstractVector{<:Distribution},
    vn::VarName,
    var,
    vi::VarInfo)

    @assert length(dists) == 1 "[Turing.assume] Turing only supports vectorizing iid distributions"

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
