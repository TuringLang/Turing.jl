@testset "emcee.jl" begin
    @testset "gdemo" begin
        Random.seed!(9876)

        n_samples = 1000
        n_walkers = 250
        
        spl = Emcee(n_walkers, 2.0)
        @test DynamicPPL.alg_str(Sampler(spl, gdemo_default)) == "Emcee"

        chain = sample(gdemo_default, spl, n_samples)
        check_gdemo(chain)

        # https://github.com/TuringLang/Turing.jl/pull/1976
        n_samples = 100_000
        chain = sample(gdemo_default, spl, n_samples)
        check_gdemo(chain)
    end

    @testset "initial parameters" begin
        nwalkers = 250
        spl = Emcee(nwalkers, 2.0)

        # No initial parameters, with im- and explicit `init_params=nothing`
        Random.seed!(1234)
        chain1 = sample(gdemo_default, spl, 1)
        Random.seed!(1234)
        chain2 = sample(gdemo_default, spl, 1; init_params=nothing)
        @test Array(chain1) == Array(chain2)

        # Initial parameters have to be specified for every walker
        @test_throws ArgumentError sample(gdemo_default, spl, 1; init_params=[2.0, 1.0])

        # Initial parameters
        chain = sample(gdemo_default, spl, 1; init_params=fill([2.0, 1.0], nwalkers))
        @test chain[:s] == fill(2.0, 1, nwalkers)
        @test chain[:m] == fill(1.0, 1, nwalkers)
    end
end
