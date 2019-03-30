struct ANUTS{AD, T} <: Hamiltonian{AD}
    n_iters   ::  Integer   # number of samples
    δ         ::  Float64   # target accept ratio
    space     ::  Set{T}    # sampling space, emtpy means all
end

"""
    ANUTS(n_iters::Integer)

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
function ANUTS{AD}(n_iters::Integer, δ::Float64=0.8, space...) where AD
    _space = isa(space, Symbol) ? Set([space]) : Set(space)
    ANUTS{AD, eltype(_space)}(n_iters, δ, _space)
end

function Sampler(alg::ANUTS{T}, s::Selector) where T <: Hamiltonian
  return Sampler(alg, Dict{Symbol,Any}(), s)
end

function sample(model::Model,
                alg::ANUTS{AD},
                n_adapt::Int=1_000,
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

    if spl.selector.tag == :default
        link!(vi, spl)
        runmodel!(model, vi, spl)
    end

    function logπ(x)::Float64
        vi[spl] = x
        return runmodel!(model, vi, spl).logp
    end


    function ∂logπ∂θ(x)::Vector{Float64}
        _, deriv = gradient_logp(x, vi, model, spl)
        return deriv
    end

    θ_init = Vector{Float64}(vi[spl])
    # Define metric space, Hamiltonian and sampling method
    metric = AdvancedHMC.DenseEuclideanMetric(θ_init)
    h = AdvancedHMC.Hamiltonian(metric, logπ, ∂logπ∂θ)
    prop = AdvancedHMC.NUTS(AdvancedHMC.Leapfrog(AdvancedHMC.find_good_eps(h, θ_init)))
    adaptor = AdvancedHMC.StanNUTSAdaptor(n_adapt,
                                          AdvancedHMC.PreConditioner(metric),
                                          AdvancedHMC.NesterovDualAveraging(spl.alg.δ, prop.integrator.ϵ))

    # Sampling
    ahmc_samples = AdvancedHMC.sample(h, prop, θ_init, spl.alg.n_iters, adaptor, n_adapt)

    for i = 1:alg.n_iters
        vi[spl] = ahmc_samples[i]
        samples[i].value = Sample(vi, spl).value
    end

    return Chain(0.0, samples)
end
