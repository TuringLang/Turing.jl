"""
    Gibbs(n_iters, algs...)

Compositional MCMC interface. Gibbs sampling combines one or more
sampling algorithms, each of which samples from a different set of
variables in a model.

Example:
```julia
@model gibbs_example(x) = begin
    v1 ~ Normal(0,1)
    v2 ~ Categorical(5)
        ...
end

# Use PG for a 'v2' variable, and use HMC for the 'v1' variable.
# Note that v2 is discrete, so the PG sampler is more appropriate
# than is HMC.
alg = Gibbs(1000, HMC(1, 0.2, 3, :v1), PG(20, 1, :v2))
```

Tips:
- `HMC` and `NUTS` are fast samplers, and can throw off particle-based
methods like Particle Gibbs. You can increase the effectiveness of particle sampling by including
more particles in the particle sampler.
"""
mutable struct Gibbs{space, A} <: AbstractGibbs
    n_iters   ::  Int     # number of Gibbs iterations
    algs      ::  A   # component sampling algorithms
    thin      ::  Bool    # if thinning to output only after a whole Gibbs sweep
    gid       ::  Int
end
function Gibbs(n_iters::Int, algs...; thin=true)
    Gibbs{buildspace(algs), typeof(algs)}(n_iters, algs, thin, 0)
end
Gibbs(alg::Gibbs, new_gid) = Gibbs(alg.n_iters, alg.algs, alg.thin, new_gid)

const GibbsComponent = Union{Hamiltonian,MH,PG}

@inline function get_gibbs_samplers(subalgs, model, n, alg, alg_str)
    if length(subalgs) == 0
        return ()
    else
        subalg = subalgs[1]
        if isa(subalg, GibbsComponent)
            return (Sampler(typeof(subalg)(subalg, n + 1 - length(subalgs)), model), get_gibbs_samplers(Base.tail(subalgs), model, n, alg, alg_str)...)
        else
            error("[$alg_str] unsupport base sampling algorithm $alg")
        end
    end
end  

function Sampler(alg::Gibbs, model::Model)
    n_samplers = length(alg.algs)
    alg_str = "Gibbs"
    samplers = get_gibbs_samplers(alg.algs, model, n_samplers, alg, alg_str)
    space = buildspace(alg.algs)
    verifyspace(space, model.pvars, alg_str)
    info = Dict{Symbol, Any}()
    info[:samplers] = samplers

    Sampler(alg, info)
end
