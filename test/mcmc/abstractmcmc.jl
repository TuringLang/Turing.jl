module TuringAbstractMCMCTests

using AbstractMCMC: AbstractMCMC
using DynamicPPL: DynamicPPL
using Random: AbstractRNG, Random
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
        return vi, nothing
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
            @test chain[1].metadata.p.vals == [0.2]
            @test DynamicPPL.getlogjoint(chain[1]) == lptrue

            # parallel sampling
            chains = sample(
                model,
                spl,
                MCMCThreads(),
                1,
                10;
                initial_params=fill(inits, 10),
                progress=false,
            )
            for c in chains
                @test c[1].metadata.p.vals == [0.2]
                @test DynamicPPL.getlogjoint(c[1]) == lptrue
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
            @test chain[1].metadata.s.vals == [4]
            @test chain[1].metadata.m.vals == [-1]
            @test DynamicPPL.getlogjoint(chain[1]) == lptrue

            # parallel sampling
            chains = sample(
                model,
                spl,
                MCMCThreads(),
                1,
                10;
                initial_params=fill(inits, 10),
                progress=false,
            )
            for c in chains
                @test c[1].metadata.s.vals == [4]
                @test c[1].metadata.m.vals == [-1]
                @test DynamicPPL.getlogjoint(c[1]) == lptrue
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
            @test !ismissing(chain[1].metadata.s.vals[1])
            @test chain[1].metadata.m.vals == [-1]

            # parallel sampling
            chains = sample(
                model,
                spl,
                MCMCThreads(),
                1,
                10;
                initial_params=fill(inits, 10),
                progress=false,
            )
            for c in chains
                @test !ismissing(c[1].metadata.s.vals[1])
                @test c[1].metadata.m.vals == [-1]
            end
        end
    end
end

@testset "Initial parameter retry logic" begin
        # Model that fails initialization first 5 times then succeeds
        init_counter_1 = Ref(0)

        @model function bad_init_model()
            init_counter_1[] += 1
            x ~ Normal(0, 1)
            Turing.@addlogprob! (init_counter_1[] > 5) ? 0.0 : -Inf
        end

        model = bad_init_model()

        @testset "NUTS sampler" begin
            Random.seed!(1234)
            chain = sample(model, NUTS(), 10)
            @test size(chain, 1) == 10
        end

        @test init_counter_1[] ≥ 6

        init_counter_2 = Ref(0)

        @model function bad_init_model_hmc()
            init_counter_2[] += 1
            x ~ Normal(0, 1)
            Turing.@addlogprob! (init_counter_2[] > 5) ? 0.0 : -Inf
        end

        @testset "HMC sampler" begin
            chain = sample(bad_init_model_hmc(), HMC(0.1, 5), 10)
            @test size(chain, 1) == 10
        end

        @test init_counter_2[] ≥ 6
        
        # Model that's impossible to initialize (always -Inf)
        @model function impossible_model()
            x ~ Normal(0, 1)
            Turing.@addlogprob! -Inf  # Always invalid
        end
        
        @testset "Impossible initialization" begin
            model = impossible_model()
            
            # Should throw an error with informative message
            @test_throws ErrorException sample(model, NUTS(), 10)
            
                sample(model, NUTS(), 10)
                @test occursin("Failed to find valid initial parameters", error_msg)
            end
        end
        
        # Model that requires many attempts
        @model function difficult_model()
            x ~ Normal(0, 50)
            # Valid only in tiny range, should trigger warning
            Turing.@addlogprob! (abs(x) < 0.05) ? 0.0 : -Inf
        end
        
        @testset "Warning at attempt 10" begin
            # Use a counter to ensure model fails exactly 30 times then succeeds
            attempt_counter = Ref(0)
            
            @model function counter_model()
                attempt_counter[] += 1
                x ~ Normal(0, 1)
                # Fail for first 30 attempts, then succeed
                Turing.@addlogprob! (attempt_counter[] > 30) ? 0.0 : -Inf
            end
            
            model = counter_model()
            
            # Should see warning at attempt 10
            @test_logs(
                (:warn, r"failed to find valid initial parameters in 10 tries"),
                match_mode=:any,
                sample(model, NUTS(), 10)
            )
            
            # Verify it actually tried more than 30 times
            @test attempt_counter[] > 30
        end
        
        @testset "Direct find_initial_params test" begin
            @model function simple_model()
                x ~ Normal(0, 1)
            end
            
            model = simple_model()
            vi = DynamicPPL.VarInfo(model)
            _, vi = DynamicPPL.init!!(model, vi, DynamicPPL.InitFromPrior())
            
            # Validator that always succeeds
            validator_success = vi -> true
            result_vi = Turing.Inference.find_initial_params(
                Random.default_rng(), model, vi, DynamicPPL.InitFromPrior(), validator_success
            )
            @test result_vi isa DynamicPPL.AbstractVarInfo
            
            # Validator that succeeds after a few tries
            counter = Ref(0)
            validator_counter = vi -> (counter[] += 1; counter[] > 3)
            result_vi = Turing.Inference.find_initial_params(
                Random.default_rng(), model, vi, DynamicPPL.InitFromPrior(), validator_counter
            )
            @test counter[] > 3
        end
end

end # module
