@testset "tempering.jl" begin
    @turing_testset "sampling" begin
        Random.seed!(0)
        N = 1000
        samplers = (
            HMC(0.1, 7),
            MH(),
            NUTS(0.65)
        )
        for s in samplers
            ts = Tempered(s, 4)
            @test typeof(ts) <: TemperedSampler
            @test typeof(ts.internal_sampler) <: typeof(s)
            
            Random.seed!(0)
            c1 = sample(gdemo_default, Tempered(s, 2), N)
            # check_gdemo(c1)
            @test c1 isa MCMCChains.Chains

            Random.seed!(0)
            c2 = sample(gdemo_default, Tempered(s, 2), N)

            @test c1.value == c2.value
        end
    end
end