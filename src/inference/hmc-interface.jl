struct HMC{AD, T} <: StaticHamiltonian{AD}
    n_iters     ::  Int       # number of samples
    ϵ           ::  Float64   # leapfrog step size
    n_leapfrog  ::  Int       # leapfrog step number
    space       ::  Set{T}    # sampling space, emtpy means all
    metricT     ::  Type{<:AHMC.AbstractMetric}
end

HMC(args...) = HMC{ADBackend()}(args...)

function HMC{AD}(
    n_iters::Int,
    ϵ::Float64,
    n_leapfrog::Int;
    metricT=AHMC.UnitEuclideanMetric
) where AD
    return HMC{AD, Any}(n_iters, ϵ, n_leapfrog, Set(), metricT)
end

function HMC{AD}(
    n_iters::Int,
    ϵ::Float64,
    n_leapfrog::Int,
    space...;
    metricT=AHMC.UnitEuclideanMetric
) where AD
    _space = isa(space, Symbol) ? Set([space]) : Set(space)
    return HMC{AD, eltype(_space)}(n_iters, ϵ, n_leapfrog, _space, metricT)
end

mutable struct HMCSampler{HMC{AD, T}} <: AbstractSampler
    alg :: HMC{AD, T}
    selector :: Selector
    vi :: VarInfo
    h :: AHMC.Hamiltonian
    traj :: AHMC.StaticTrajectory
    adaptor :: AHMC.StanNUTSAdaptor
    eval_num :: Int
    i :: Int
end

function sample_init!(
    rng::AbstractRNG,
    ℓ::ModelType,
    s::HMCSampler,
    N::Integer;
    save_state::Bool=false,
    resume_from::Union{Nothing, MCMCChains.Chains}=nothing,
    reuse_spl_n::Int=0,
    adaptor::AdaptorType,
    init_theta::Union{Nothing,Array{<:Any,1}}=nothing,
    kwargs...
) where {ModelType<:Sampleable, AdaptorType<:AHMC.AbstractAdaptor}
    # If there's a sampler passed in, use that instead.
    s = reuse_spl_n > 0 ?
        resume_from.info[:spl] :
        s

    # Catch bad sampler passes.
    @assert isa(s.alg, Hamiltonian)
        "[Turing] alg type mismatch; please use resume() to re-use spl"

    # Resume the selector.
    resume_from != nothing && (s.selector = resume_from.info[:spl].selector)

    # Create VarInfo
    s.vi = if resume_from == nothing
        vi_ = VarInfo()
        runmodel!(ℓ, vi_, SampleFromUniform())
        vi_
    else
        deepcopy(resume_from.info[:vi])
    end

    # Get `init_theta`
    if init_theta != nothing
        @info "Using passed-in initial variable values" init_theta
        # Convert individual numbers to length 1 vector
        init_theta = [size(v) == () ? [v] : v for v in init_theta]
        # Flatten `init_theta`
        init_theta_flat = foldl(vcat, map(vec, init_theta))
        # Create a mask to inidicate which values are not missing
        theta_mask = map(x -> !ismissing(x), init_theta_flat)
        # Get all values
        theta = vi[spl]
        # Update those which are provided (i.e. not missing)
        theta[theta_mask] .= init_theta_flat[theta_mask]
        # Update in `vi`
        s.vi[spl] = theta
    end

    # Convert to transformed space
    if spl.selector.tag == :default
        link!(s.vi, s)
        runmodel!(ℓ, s.vi, s)
    end

    # Init h, prop and adaptor
    step(ℓ, s, s.vi, Val(true); adaptor=adaptor)
end

function Sampler(alg::HMC{AD, T}, s::Selector=Selector()) where {AD, T}
    HMCSampler{HMC{AD, T}}(alg, s, )
end

end
