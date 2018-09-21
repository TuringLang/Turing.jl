# Concrete algorithm implementations.
include("support/helper.jl")
include("support/resample.jl")
# @suppress_err begin
include("support/distributions.jl")
# end
include("support/hmc_core.jl")
include("support/adapt.jl")
include("support/init.jl")
include("support/stan-interface.jl")
include("hmcda.jl")
include("nuts.jl")
include("sghmc.jl")
include("sgld.jl")
include("hmc.jl")
include("mh.jl")
include("is.jl")
include("smc.jl")
include("pgibbs.jl")
include("pmmh.jl")
include("ipmcmc.jl")
include("gibbs.jl")

## Fallback functions

# utility funcs for querying sampler information
require_gradient(spl::Sampler) = false
require_particles(spl::Sampler) = false

assume(spl::Sampler, dist::Distribution) =
error("Turing.assume: unmanaged inference algorithm: $(typeof(spl))")

observe(spl::Sampler, weight::Float64) =
error("Turing.observe: unmanaged inference algorithm: $(typeof(spl))")

## Default definitions for assume, observe, when sampler = nothing.
##  Note: `A<:Union{Nothing, SampleFromPrior, HamiltonianRobustInit}` can be
##   simplified into `A<:Union{SampleFromPrior, HamiltonianRobustInit}` when
##   all `spl=nothing` is replaced with `spl=SampleFromPrior`.
function assume(spl::A,
    dist::Distribution,
    vn::VarName,
    vi::VarInfo) where {A<:Union{Nothing, SampleFromPrior, HamiltonianRobustInit}}

    if haskey(vi, vn)
        r = vi[vn]
    else
        r = isa(spl, HamiltonianRobustInit) ? init(dist) : rand(dist)
        push!(vi, vn, r, dist, 0)
    end
    # NOTE: The importance weight is not correctly computed here because
    #       r is genereated from some uniform distribution which is different from the prior
    # acclogp!(vi, logpdf_with_trans(dist, r, istrans(vi, vn)))

    r, logpdf_with_trans(dist, r, istrans(vi, vn))

end

function assume(spl::A,
    dists::Vector{T},
    vn::VarName,
    var::Any,
    vi::VarInfo) where {T<:Distribution, A<:Union{Nothing, SampleFromPrior, HamiltonianRobustInit}}

    @assert length(dists) == 1 "Turing.assume only support vectorizing i.i.d distribution"
    dist = dists[1]
    n = size(var)[end]

    vns = map(i -> copybyindex(vn, "[$i]"), 1:n)

    if haskey(vi, vns[1])
        rs = vi[vns]
    else
        rs = isa(spl, HamiltonianRobustInit) ? init(dist, n) : rand(dist, n)

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

    var, sum(logpdf_with_trans(dist, rs, istrans(vi, vns[1])))

end

function observe(spl::A,
    dist::Distribution,
    value::Any,
    vi::VarInfo) where {A<:Union{Nothing, SampleFromPrior, HamiltonianRobustInit}}

    vi.num_produce += 1
    @debug "dist = $dist"
    @debug "value = $value"

    # acclogp!(vi, logpdf(dist, value))
    logpdf(dist, value)

end

function observe(spl::A,
    dists::Vector{T},
    value::Any,
    vi::VarInfo) where {T<:Distribution, A<:Union{Nothing, SampleFromPrior, HamiltonianRobustInit}}

    @assert length(dists) == 1 "Turing.observe only support vectorizing i.i.d distribution"
    dist = dists[1]
    @assert isa(dist, UnivariateDistribution) || isa(dist, MultivariateDistribution) "Turing.observe: vectorizing matrix distribution is not supported"
    if isa(dist, UnivariateDistribution)  # only univariate distributions support broadcast operation (logpdf.) by Distributions.jl
        # acclogp!(vi, sum(logpdf.(Ref(dist), value)))
        sum(logpdf.(Ref(dist), value))
    else
        # acclogp!(vi, sum(logpdf(dist, value)))
        sum(logpdf(dist, value))
    end

end
