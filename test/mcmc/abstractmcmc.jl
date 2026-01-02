module TuringAbstractMCMCTests

using AbstractMCMC: AbstractMCMC
using DynamicPPL: DynamicPPL
using Random: AbstractRNG
using Test: @test, @testset, @test_throws
using Turing

@testset "Disabling check_model" begin
    # Set up a model for which check_model errors.
    @model f() = x ~ Normal()
    model = f()
    Turing.Inference._check_model(::typeof(model)) = error("nope")
    # Make sure that default sampling does throw the error.
    @test_throws "nope" sample(model, NUTS(), 100)
    @test_throws "nope" sample(model, NUTS(), MCMCThreads(), 100, 2)
    @test_throws "nope" sample(model, NUTS(), MCMCSerial(), 100, 2)
    @test_throws "nope" sample(model, NUTS(), MCMCDistributed(), 100, 2)
    # Now disable the check and make sure sampling works.
    @test sample(model, NUTS(), 100; check_model=false) isa Any
    @test sample(model, NUTS(), MCMCThreads(), 100, 2; check_model=false) isa Any
    @test sample(model, NUTS(), MCMCSerial(), 100, 2; check_model=false) isa Any
    @test sample(model, NUTS(), MCMCDistributed(), 100, 2; check_model=false) isa Any
end

@testset "Initial parameters" begin
    # Dummy algorithm that just returns initial value and does not perform any sampling
    abstract type OnlyInit <: AbstractMCMC.AbstractSampler end
    struct OnlyInitDefault <: OnlyInit end
    struct OnlyInitUniform <: OnlyInit end
    Turing.Inference.init_strategy(::OnlyInitUniform) = InitFromUniform()
    function Turing.Inference.initialstep(
        rng::AbstractRNG,
        model::DynamicPPL.Model,
        ::OnlyInit,
        vi::DynamicPPL.VarInfo=DynamicPPL.VarInfo(rng, model);
        kwargs...,
    )
        return DynamicPPL.ParamsWithStats(vi, model), nothing
    end

    @testset "init_strategy" begin
        # check that the default init strategy is prior
        @test Turing.Inference.init_strategy(OnlyInitDefault()) == InitFromPrior()
        @test Turing.Inference.init_strategy(OnlyInitUniform()) == InitFromUniform()
    end

    for spl in (OnlyInitDefault(), OnlyInitUniform())
        # model with one variable: initialization p = 0.2
        @model function coinflip()
            p ~ Beta(1, 1)
            return 10 ~ Binomial(25, p)
        end
        model = coinflip()
        lptrue = logpdf(Binomial(25, 0.2), 10)
        let inits = InitFromParams((; p=0.2))
            chain = sample(model, spl, 1; initial_params=inits, progress=false)
            @test only(chain[@varname(p)]) == 0.2
            @test only(chain[:logjoint]) == lptrue

            # parallel sampling
            c = sample(
                model,
                spl,
                MCMCThreads(),
                1,
                10;
                initial_params=fill(inits, 10),
                progress=false,
            )
            @test all(c[@varname(p)] .== 0.2)
            @test all(c[:logjoint] .== lptrue)
        end

        # check that Vector no longer works
        @test_throws ArgumentError sample(
            model, spl, 1; initial_params=[4, -1], progress=false
        )
        @test_throws ArgumentError sample(
            model, spl, 1; initial_params=[missing, -1], progress=false
        )

        # model with two variables: initialization s = 4, m = -1
        @model function twovars()
            s ~ InverseGamma(2, 3)
            return m ~ Normal(0, sqrt(s))
        end
        model = twovars()
        lptrue = logpdf(InverseGamma(2, 3), 4) + logpdf(Normal(0, 2), -1)
        for inits in (
            InitFromParams((s=4, m=-1)),
            (s=4, m=-1),
            InitFromParams(Dict(@varname(s) => 4, @varname(m) => -1)),
            Dict(@varname(s) => 4, @varname(m) => -1),
        )
            chain = sample(model, spl, 1; initial_params=inits, progress=false)
            @test only(chain[@varname(s)]) == 4
            @test only(chain[@varname(m)]) == -1
            @test only(chain[:logjoint]) == lptrue

            # parallel sampling
            c = sample(
                model,
                spl,
                MCMCThreads(),
                1,
                10;
                initial_params=fill(inits, 10),
                progress=false,
            )
            @test all(c[@varname(s)] .== 4)
            @test all(c[@varname(m)] .== -1)
            @test all(c[:logjoint] .== lptrue)
        end

        # set only m = -1
        for inits in (
            InitFromParams((; s=missing, m=-1)),
            InitFromParams(Dict(@varname(s) => missing, @varname(m) => -1)),
            (; s=missing, m=-1),
            Dict(@varname(s) => missing, @varname(m) => -1),
            InitFromParams((; m=-1)),
            InitFromParams(Dict(@varname(m) => -1)),
            (; m=-1),
            Dict(@varname(m) => -1),
        )
            chain = sample(model, spl, 1; initial_params=inits, progress=false)
            @test haskey(chain, @varname(s))
            @test only(chain[@varname(s)]) isa Real
            @test only(chain[@varname(m)]) == -1

            # parallel sampling
            c = sample(
                model,
                spl,
                MCMCThreads(),
                1,
                10;
                initial_params=fill(inits, 10),
                progress=false,
            )
            @test haskey(c, @varname(s))
            @test all(x -> x isa Real, c[@varname(s)])
            @test all(c[@varname(m)] .== -1)
        end
    end
end

end # module
