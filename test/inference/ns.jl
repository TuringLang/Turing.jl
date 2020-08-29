#using Turing, Random, Test
#import Turing.Inference
#import NestedSamplers

#dir = splitdir(splitdir(pathof(Turing))[1])[1]
#include(dir*"/test/test_utils/AllUtils.jl")

#@testset "ns.jl" begin
#    @testset "gdemo" begin
#        Random.seed!(1729)
        
#        N = 1000
#        ndims = 3
#        nactive = 100
#        bound_type = NestedSamplers.Bounds.Ellipsoid
#        proposal_type = NestedSamplers.Proposals.Uniform()
        
#        spl = NS(ndims, nactive, bound_type, proposal_type)
#        @test DynamicPPL.alg_str(Sampler(spl, gdemo_default)) == "NS"
        
#        chain = sample(gdemo_default, spl, N)
#        check_gdemo(chain)
#    end
#end
