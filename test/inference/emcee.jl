using Turing, Random, Test
import Turing.Inference
import AdvancedMH


@testset "emcee.jl" begin
    @testset "gdemo" begin
        Random.seed!(9876)

        n_samples = 1000
        n_walkers = 250
        
        spl = Emcee(n_walkers, 2.0)
        @test DynamicPPL.alg_str(Sampler(spl, gdemo_default)) == "Emcee"

        chain = sample(gdemo_default, spl, n_samples)
        check_gdemo(chain)
    end
end
