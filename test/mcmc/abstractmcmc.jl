module TuringAbstractMCMCTests

using AbstractMCMC: AbstractMCMC
using DynamicPPL: DynamicPPL
using LogDensityProblems: LogDensityProblems
using Random: AbstractRNG, Random, Xoshiro
using Test: @test, @testset, @test_throws, @test_logs
using Turing
using Logging

@testset "Disabling check_model" begin
    # Set up a model for which check_model errors.
    @model f() = x ~ Normal()
    model = f()
    spl = NUTS()
    Turing._check_model(::typeof(model), ::typeof(spl)) = error("nope")
    # Make sure that default sampling does throw the error.
    @test_throws "nope" sample(model, spl, 10)
    @test_throws "nope" sample(model, spl, MCMCThreads(), 10, 2)
    @test_throws "nope" sample(model, spl, MCMCSerial(), 10, 2)
    @test_throws "nope" sample(model, spl, MCMCDistributed(), 10, 2)
    # Now disable the check and make sure sampling works.
    @test sample(model, spl, 10; check_model=false) isa Any
    @test sample(model, spl, MCMCThreads(), 10, 2; check_model=false) isa Any
    @test sample(model, spl, MCMCSerial(), 10, 2; check_model=false) isa Any
    @test sample(model, spl, MCMCDistributed(), 10, 2; check_model=false) isa Any
end

@testset "find_initial_params_ldf" begin
    @testset "basic interface" begin
        @model function normal_model()
            x ~ Normal(0, 1)
            return y ~ Normal(x, 1)
        end
        @testset for init_strategy in
                     (InitFromPrior(), InitFromUniform(), InitFromParams((x=0.5, y=-0.3)))
            model = normal_model()
            ldf = DynamicPPL.LogDensityFunction(
                model, DynamicPPL.getlogjoint_internal, DynamicPPL.LinkAll()
            )
            rng = Xoshiro(468)
            x = Turing.Inference.find_initial_params_ldf(rng, ldf, init_strategy)
            @test x isa AbstractVector{<:Real}
            @test length(x) == LogDensityProblems.dimension(ldf)
        end
    end

    @testset "warning for difficult init params" begin
        attempt = 0
        @model function demo_warn_initial_params()
            x ~ Normal()
            if (attempt += 1) < 30
                @addlogprob! -Inf
            end
        end
        ldf = DynamicPPL.LogDensityFunction(
            demo_warn_initial_params(),
            DynamicPPL.getlogjoint_internal,
            DynamicPPL.LinkAll(),
        )
        @test_logs (:warn, r"consider providing a different initialisation strategy") Turing.Inference.find_initial_params_ldf(
            Xoshiro(468), ldf, InitFromUniform()
        )
    end

    @testset "errors after max_attempts" begin
        @model function impossible_model()
            x ~ Normal()
            @addlogprob! -Inf
        end
        model = impossible_model()
        ldf = DynamicPPL.LogDensityFunction(
            model, DynamicPPL.getlogjoint_internal, DynamicPPL.LinkAll()
        )
        @test_throws ErrorException Turing.Inference.find_initial_params_ldf(
            Xoshiro(468), ldf, InitFromUniform()
        )
    end
end

@testset "Initial parameters" begin
    # Dummy algorithm that just returns initial value and does not perform any sampling
    abstract type OnlyInit <: AbstractMCMC.AbstractSampler end
    struct OnlyInitDefault <: OnlyInit end
    struct OnlyInitUniform <: OnlyInit end
    Turing.Inference.init_strategy(::OnlyInitUniform) = InitFromUniform()
    function AbstractMCMC.step(
        rng::AbstractRNG,
        model::DynamicPPL.Model,
        ::OnlyInit;
        initial_params::DynamicPPL.AbstractInitStrategy,
        kwargs...,
    )
        accs = DynamicPPL.OnlyAccsVarInfo()
        accs = DynamicPPL.setacc!!(accs, DynamicPPL.RawValueAccumulator(false))
        _, accs = DynamicPPL.init!!(
            rng, model, accs, initial_params, DynamicPPL.UnlinkAll()
        )
        return accs, nothing
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
            oavis = sample(model, spl, 1; initial_params=inits, progress=false)
            oavi = only(oavis)
            @test DynamicPPL.get_raw_values(oavi)[@varname(p)] == 0.2
            @test DynamicPPL.getlogjoint(oavi) == lptrue

            # parallel sampling
            chn = sample(
                model,
                spl,
                MCMCThreads(),
                1,
                10;
                initial_params=fill(inits, 10),
                progress=false,
            )
            for c in chains
                oavi = only(c)
                @test DynamicPPL.get_raw_values(oavi)[@varname(p)] == 0.2
                @test DynamicPPL.getlogjoint(oavi) == lptrue
            end
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
            oavi = only(chain)
            vnt = DynamicPPL.get_raw_values(oavi)
            @test vnt[@varname(s)] == 4
            @test vnt[@varname(m)] == -1
            @test DynamicPPL.getlogjoint(oavi) == lptrue

            # parallel sampling
            chn = sample(
                model,
                spl,
                MCMCThreads(),
                1,
                10;
                initial_params=fill(inits, 10),
                progress=false,
            )
            for c in chains
                oavi = only(c)
                vnt = DynamicPPL.get_raw_values(oavi)
                @test vnt[@varname(s)] == 4
                @test vnt[@varname(m)] == -1
                @test DynamicPPL.getlogjoint(oavi) == lptrue
            end
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
            vnt = DynamicPPL.get_raw_values(only(chain))
            @test !ismissing(vnt[@varname(s)])
            @test vnt[@varname(m)] == -1

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
            for c in chains
                vnt = DynamicPPL.get_raw_values(only(c))
                @test !ismissing(vnt[@varname(s)])
                @test vnt[@varname(m)] == -1
            end
        end
    end
end

@testset "post_sample_hook" begin
    @model function f()
        x ~ Normal()
        if x < 0
            @addlogprob! -Inf
            return nothing
        end
    end
    warn_message = r"There were \d+ divergent transitions"
    for spl in [NUTS(), HMC(0.1, 5), HMCDA(200, 0.65, 0.3)]
        @test_logs min_level = Logging.Warn match_mode = :any (:warn, warn_message) sample(
            f(), spl, 1000
        ),
        @test_logs min_level = Logging.Warn match_mode = :any (:warn, warn_message) sample(
            f(), spl, MCMCThreads(), 1000, 2
        )
    end
end

end # module
