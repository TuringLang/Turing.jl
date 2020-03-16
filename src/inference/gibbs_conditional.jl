# @model test(y) = begin
    # p ~ Uniform()
    # y ~ Bernoulli(p)
# end


struct GibbsConditional{space, C} <: InferenceAlgorithm 
    conditional::C

    function GibbsConditional(sym::Symbol, conditional::C) where {C}
        return new{sym, C}(conditional)
    end
end

getspace(::GibbsConditional{S}) where {S} = (S,)
alg_str(::Sampler{<:GibbsConditional}) = "GibbsConditional"


#################
# Transition #
#################

struct GibbsConditionalTransition{T, F<:AbstractFloat}
    θ::T
    lp::F
end

function GibbsConditionalTransition(spl::Sampler{<:GibbsConditional}, lp)
    θ = tonamedtuple(spl.state.vi)
    return GibbsConditionalTransition(θ, mh_trans.lp)
end


mutable struct GibbsConditionalState{V<:VarInfo} <: AbstractSamplerState
    vi::V
    # density_model::AMH.DensityModel
end

function Sampler(
    alg::GibbsConditional,
    model::Model,
    s::Selector=Selector()
)
    # Set up info dict.
    info = Dict{Symbol, Any}()

    # Make a varinfo.
    vi = VarInfo(model)

    # Set up state struct.
    state = GibbsConditionalState(vi)

    # Generate a sampler.
    spl = Sampler(alg, info, s, state)

    return spl
end

function AbstractMCMC.sample_init!(
    rng::AbstractRNG,
    model::Model,
    spl::Sampler{<:GibbsConditional},
    N::Integer;
    verbose::Bool=true,
    resume_from=nothing,
    kwargs...
)
    # Resume the sampler.
    set_resume!(spl; resume_from=resume_from, kwargs...)

    # Get `init_theta`
    initialize_parameters!(spl; verbose=verbose, kwargs...)
end

function AbstractMCMC.step!(
    rng::AbstractRNG,
    model::Model,
    spl::Sampler{<:GibbsConditional{S}},
    N::Integer,
    transition;
    kwargs...
) where {S}
    if spl.selector.rerun # Recompute joint in logp
        runmodel!(model, spl.state.vi)
    end

    vn = VarName{S}("")
    condvals = getcondvals(spl.state.vi, vn)
    conddist = spl.alg.conditional(condvals)
    updated = rand(rng, conddist)
    spl.state.vi[vn] = [updated]

    return Transition(spl)
end


function getcondvals(vi::VarInfo, vn::VarName)
    conditionals = filter(!(==(vn)), getallvns(vi))
    condnames = Tuple(getsym.(conditionals))
    condvals = getindex.(getval.(Ref(vi), conditionals), 1)  # getval returns a view slice!
    return NamedTuple{condnames}(condvals)
end

getallvns(vi::UntypedVarInfo) = vi.metadata.vns
getallvns(vi::TypedVarInfo) = foldl(vcat, m.vns for m in values(vi.metadata); init=VarName[])


####
#### Compiler interface, i.e. tilde operators.
####
function assume(
    spl::Sampler{<:GibbsConditional},
    dist::Distribution,
    vn::VarName,
    vi::VarInfo
)
    updategid!(vi, vn, spl)
    r = vi[vn]
    return r, logpdf_with_trans(dist, r, istrans(vi, vn))
end

function dot_assume(
    spl::Sampler{<:GibbsConditional},
    dist::MultivariateDistribution,
    vn::VarName,
    var::AbstractMatrix,
    vi::VarInfo,
)
    @assert dim(dist) == size(var, 1)
    getvn = i -> VarName(vn, vn.indexing * "[:,$i]")
    vns = getvn.(1:size(var, 2))
    updategid!.(Ref(vi), vns, Ref(spl))
    r = vi[vns]
    var .= r
    return var, sum(logpdf_with_trans(dist, r, istrans(vi, vns[1])))
end
function dot_assume(
    spl::Sampler{<:GibbsConditional},
    dists::Union{Distribution, AbstractArray{<:Distribution}},
    vn::VarName,
    var::AbstractArray,
    vi::VarInfo,
)
    getvn = ind -> VarName(vn, vn.indexing * "[" * join(Tuple(ind), ",") * "]")
    vns = getvn.(CartesianIndices(var))
    updategid!.(Ref(vi), vns, Ref(spl))
    r = reshape(vi[vec(vns)], size(var))
    var .= r
    return var, sum(logpdf_with_trans.(dists, r, istrans(vi, vns[1])))
end

function observe(
    spl::Sampler{<:GibbsConditional},
    d::Distribution,
    value,
    vi::VarInfo,
)
    return observe(SampleFromPrior(), d, value, vi)
end

function dot_observe(
    spl::Sampler{<:GibbsConditional},
    ds::Union{Distribution, AbstractArray{<:Distribution}},
    value::AbstractArray,
    vi::VarInfo,
)
    return dot_observe(SampleFromPrior(), ds, value, vi)
end
