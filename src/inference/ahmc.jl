struct ANUTS{AD, T} <: Hamiltonian{AD}
    n_iters   ::  Int   # number of samples
    δ         ::  Float64   # target accept ratio
    max_depth ::  Int
    Δ_max     ::  Float64
    space     ::  Set{T}    # sampling space, emtpy means all
end

"""
    ANUTS(n_iters::Int)

Dynamic No U-Turn Sampling algorithm provided by the AdvancedHMC package.

```julia
using Turing

# Model definition.
@model gdemo(x, y) = begin
  s ~ InverseGamma(2,3)
  m ~ Normal(0,sqrt(s))
  x ~ Normal(m, sqrt(s))
  y ~ Normal(m, sqrt(s))
  return s, m
end

# Pull 2,000 samples using ANUTS.
chn = sample(gdemo(1.5, 2.0), ANUTS(2_000))
```
"""
ANUTS(args...) = ANUTS{ADBackend()}(args...)
function ANUTS{AD}(n_iters::Int, δ::Float64=0.8, max_depth::Int=10, Δ_max::Float64=1_000.0, space...) where AD
    _space = isa(space, Symbol) ? Set([space]) : Set(space)
    ANUTS{AD, eltype(_space)}(n_iters, δ, max_depth, Δ_max, _space)
end

function Sampler(alg::ANUTS{T}, s::Selector) where T <: Hamiltonian
  return Sampler(alg, Dict{Symbol,Any}(), s)
end

function sample(model::Model,
                alg::ANUTS{AD},
                n_adapt::Int=1_000;
                rng::AbstractRNG=GLOBAL_RNG,
                metric_type=AdvancedHMC.DenseEuclideanMetric,
                init_theta::Union{Nothing,Array{<:Any,1}}=nothing,
                init_eps::Union{Nothing,Float64}=nothing,
                ) where AD

    spl = Sampler(alg)

    n = alg.n_iters
    samples = Array{Sample}(undef, n)
    weight = 1 / n
    for i = 1:n
        samples[i] = Sample(weight, Dict{Symbol, Any}())
    end

    vi = VarInfo()
    model(vi, SampleFromUniform())
    if init_theta != nothing
        println("Using init_theta=$init_theta")
        init_theta_flat = foldl(vcat, init_theta)
        theta_mask = map(x -> !ismissing(x), init_theta_flat)
        theta = vi[spl]
        theta[theta_mask] .= init_theta_flat[theta_mask]
        vi[spl] = theta
    end

    if spl.selector.tag == :default
        link!(vi, spl)
        runmodel!(model, vi, spl)
    end

    function logπ(x)::Float64
        x_old, lj_old = vi[spl], vi.logp
        vi[spl] = x
        runmodel!(model, vi, spl).logp
        lj = vi.logp
        vi[spl] = x_old
        setlogp!(vi, lj_old)
        return lj
    end

    function ∂logπ∂θ(x)::Vector{Float64}
        x_old, lj_old = vi[spl], vi.logp
        _, deriv = gradient_logp(x, vi, model, spl)
        vi[spl] = x_old
        setlogp!(vi, lj_old)
        return deriv
    end

    θ_init = Vector{Float64}(vi[spl])
    # Define metric space, Hamiltonian and sampling method
    metric = metric_type(length(θ_init))
    h = AdvancedHMC.Hamiltonian(metric, logπ, ∂logπ∂θ)
    if init_eps == nothing
        init_eps = AdvancedHMC.find_good_eps(h, θ_init)
        @info "Found initial step size" init_eps
    end
    prop = AdvancedHMC.NUTS(AdvancedHMC.Leapfrog(init_eps), spl.alg.max_depth, spl.alg.Δ_max)
    adaptor = AdvancedHMC.StanNUTSAdaptor(n_adapt,
                                          AdvancedHMC.PreConditioner(metric),
                                          AdvancedHMC.NesterovDualAveraging(spl.alg.δ, prop.integrator.ϵ))

    # Sampling
    ahmc_samples = AdvancedHMC.sample(rng, h, prop, θ_init, spl.alg.n_iters, adaptor, n_adapt)

    for i = 1:alg.n_iters
        vi[spl] = ahmc_samples[i]
        samples[i].value = Sample(vi, spl).value
    end

    return Chain(0.0, samples)
end
