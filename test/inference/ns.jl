using Turing, Random, Test
import Turing.Inference

dir = splitdir(splitdir(pathof(Turing))[1])[1]
include(dir*"/test/test_utils/AllUtils.jl")

@testset "ns.jl" begin
    @turing_testset "NS constructor" begin
        Random.seed!(1729)
        N = 1000
        s = NS()
        @test DynamicPPL.alg_str(Sampler(s, gdemo_default)) == "NS"
        c = sample(gdemo_default, s, N)
        
        ## within Gibbs as well ??
        ## s1 = Gibbs(NS(:m), NS(:s))
        ## c1 = sample(gdemo_default, s1, N)
    end
    
    @numerical_testset "NS inference" begin
        Random.seed!(729)
        alg = NS()
        chain = sample(gdemo_default, alg, 1500)
        check_gdemo(chain, atol = 0.1)

        ## within Gibbs as well ??
        ## NS within Gibbs
        ## alg = Gibbs(NS(:m), NS(:s))
        ## chain = sample(gdemo_default, alg, 1500)
        ## check_gdemo(chain, atol = 0.1)
    end
end
