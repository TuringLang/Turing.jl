###
### DynamicHMC backend - https://github.com/tpapp/DynamicHMC.jl
###

struct DynamicNUTS{AD, T} <: Hamiltonian{AD}
    n_iters   ::  Integer   # number of samples
    space     ::  Set{T}    # sampling space, emtpy means all
end

"""
    DynamicNUTS(n_iters::Integer)

Dynamic No U-Turn Sampling algorithm provided by the DynamicHMC package.
To use it, make sure you have the DynamicHMC package installed.

"""
DynamicNUTS(args...) = DynamicNUTS{ADBackend()}(args...)
function DynamicNUTS{AD}(n_iters::Integer, space...) where AD
    _space = isa(space, Symbol) ? Set([space]) : Set(space)
    DynamicNUTS{AD, eltype(_space)}(n_iters, _space)
end

function Sampler(alg::DynamicNUTS{T}, s::Selector=Selector()) where T <: Hamiltonian
  return Sampler(alg, Dict{Symbol,Any}(), s)
end

function sample(model::Model,
                alg::DynamicNUTS{AD},
                ) where AD

    spl = Sampler(alg)

    n = alg.n_iters
    samples = Array{Sample}(undef, n)
    weight = 1 / n
    for i = 1:n
        samples[i] = Sample(weight, Dict{Symbol, Any}())
    end

    vi = VarInfo(model)
    runmodel!(model, vi, SampleFromUniform())

    if spl.selector.tag == :default
        link!(vi, spl)
        runmodel!(model, vi, spl)
    end

    function _lp(x)
        value, deriv = gradient_logp(x, vi, model, spl)
        return ValueGradient(value, deriv)
    end

    chn_dynamic, _ = NUTS_init_tune_mcmc(FunctionLogDensity(length(vi[spl]), _lp), alg.n_iters)

    for i = 1:alg.n_iters
        vi[spl] = chn_dynamic[i].q
        samples[i].value = Sample(vi, spl).value
    end

    return Chain(0.0, samples)
end
