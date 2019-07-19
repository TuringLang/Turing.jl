###
### DynamicHMC backend - https://github.com/tpapp/DynamicHMC.jl
###
struct DynamicNUTS{AD, space} <: Hamiltonian{AD} end

"""
    DynamicNUTS()

Dynamic No U-Turn Sampling algorithm provided by the DynamicHMC package.
To use it, make sure you have the DynamicHMC package installed.

"""
DynamicNUTS(args...) = DynamicNUTS{ADBackend()}(args...)
DynamicNUTS{AD}() where AD = DynamicNUTS{AD, ()}()
function DynamicNUTS{AD}(space::Symbol...) where AD
    DynamicNUTS{AD, space}()
end

getspace(::Type{<:DynamicNUTS{<:Any, space}}) where {space} = space
getspace(alg::DynamicNUTS{<:Any, space}) where {space} = space

function sample(
    rng::AbstractRNG,
    model::Model,
    spl::Sampler{<:DynamicNUTS},
    N::Integer;
    kwargs...
)
    samples = Array{Transition}(undef, N)

    runmodel!(model, spl.state.vi, SampleFromUniform())

    if spl.selector.tag == :default
        link!(spl.state.vi, spl)
        runmodel!(model, spl.state.vi, spl)
    end

    function _lp(x)
        value, deriv = gradient_logp(x, spl.state.vi, model, spl)
        return ValueGradient(value, deriv)
    end

    chn_dynamic, _ = NUTS_init_tune_mcmc(
        FunctionLogDensity(
            length(spl.state.vi[spl]),
            _lp
        ),
        N
    )

    for i = 1:N
        spl.state.vi[spl] = chn_dynamic[i].q
        samples[i] = transition(spl)
    end

    return Chains(rng, model, spl, N, samples; kwargs...)
end

function Sampler(
    alg::DynamicNUTS{AD},
    model::Turing.Model,
    s::Selector=Selector()
) where AD
    return Sampler(alg, Dict{Symbol,Any}(), s, BlankState(VarInfo(model)))
end
