module CallbacksTests

using Test, Turing, AbstractMCMC, Random, Distributions, LinearAlgebra
using Turing: DynamicPPL

struct UnmarkedMH <: AbstractMCMC.AbstractSampler end

function AbstractMCMC.step(
    rng::Random.AbstractRNG, model::DynamicPPL.Model, ::UnmarkedMH, state...; kwargs...
)
    return AbstractMCMC.step(rng, model, MH(), state...; kwargs...)
end

@model function test_normals()
    x ~ Normal()
    return y ~ MvNormal(zeros(3), I)
end

@testset "AbstractMCMC Callbacks Interface" begin
    model = test_normals()

    samplers = [
        ("NUTS", NUTS(10, 0.65)),
        ("HMC", HMC(0.1, 5)),
        ("MH", MH()),
        ("ESS", ESS()),
        ("Gibbs", Gibbs(:x => HMC(0.1, 5), :y => MH())),
        ("SGHMC", SGHMC(; learning_rate=0.01, momentum_decay=1e-2)),
        ("PG", PG(10)),
    ]

    for (name, sampler) in samplers
        @testset "$name" begin
            t1, s1 = AbstractMCMC.step(
                Random.default_rng(),
                model,
                sampler;
                initial_params=Turing.Inference.init_strategy(sampler),
            )

            # ParamsWithStats returns named params (not θ[i])
            pws = AbstractMCMC.ParamsWithStats(
                model, sampler, t1, s1; params=true, stats=true
            )
            pairs_dict = Dict(k => v for (k, v) in Base.pairs(pws))
            # Keys are Symbols since ParamsWithStats stores NamedTuple internally
            @test haskey(pairs_dict, Symbol("x"))
            @test haskey(pairs_dict, Symbol("y"))
            @test pairs_dict[Symbol("y")] isa AbstractVector
            @test length(pairs_dict[Symbol("y")]) == 3

            # Check stats contain lp
            @test haskey(pairs_dict, :lp) || haskey(pairs_dict, :logjoint)
        end
    end

    # NUTS second step has full AHMC transition metrics
    @testset "NUTS Transition Metrics" begin
        sampler = NUTS(10, 0.65)
        rng = Random.default_rng()
        t1, s1 = AbstractMCMC.step(
            rng, model, sampler; initial_params=Turing.Inference.init_strategy(sampler)
        )
        t2, s2 = AbstractMCMC.step(rng, model, sampler, s1)

        pws = AbstractMCMC.ParamsWithStats(model, sampler, t2, s2; params=true, stats=true)
        pairs_dict = Dict(k => v for (k, v) in Base.pairs(pws))

        # Keys are Symbols from NamedTuple
        @test haskey(pairs_dict, :tree_depth)
        @test haskey(pairs_dict, :n_steps)
        @test haskey(pairs_dict, :acceptance_rate)
        @test haskey(pairs_dict, :hamiltonian_energy)
    end
end

@testset "Varying-dimension checks" begin
    @model function dynamic()
        x ~ Normal()
        if x > 0
            z ~ Normal()
        end
    end

    function sample_dynamic(seed, sampler=UnmarkedMH(); n=2, kwargs...)
        return sample(Xoshiro(seed), dynamic(), sampler, n; progress=false, kwargs...)
    end

    @test_throws r"z appeared or disappeared.*UnmarkedMH" sample_dynamic(
        4; initial_params=InitFromParams((; x=-1.0))
    )
    @test_throws r"z appeared or disappeared.*UnmarkedMH" sample_dynamic(
        1; initial_params=InitFromParams((; x=1.0, z=0.0))
    )

    # The change occurs between the discarded initial draw and the first retained draw.
    @test_throws r"z appeared or disappeared.*UnmarkedMH" sample_dynamic(
        4; n=1, initial_params=InitFromParams((; x=-1.0)), num_warmup=1, discard_initial=1
    )

    @test_throws r"z appeared or disappeared.*UnmarkedMH" sample_dynamic(
        4, RepeatSampler(UnmarkedMH(), 1), initial_params=InitFromParams((; x=-1.0))
    )

    callback_calls = Ref(0)
    callback(args...; kwargs...) = (callback_calls[] += 1)
    @model fixed() = x ~ Normal()
    sample(Xoshiro(4), fixed(), UnmarkedMH(), 3; callback, progress=false)
    @test callback_calls[] == 3

    @test Turing.Inference.allow_varying_dimension(Prior())
    @test Turing.Inference.allow_varying_dimension(PG(5))
    @test sample_dynamic(4, PG(5); n=10) isa VNChain
end

end
