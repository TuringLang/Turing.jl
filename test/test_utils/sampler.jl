module SamplerTestUtils

using Turing
using Test

"""
Check that when sampling with `spl`, the resulting chain contains log-density
metadata that is correct.
"""
function test_chain_logp_metadata(spl)
    @model function f()
        # some prior term (but importantly, one that is constrained, i.e., can
        # be linked with non-identity transform)
        x ~ LogNormal()
        # some likelihood term
        return 1.0 ~ Normal(x)
    end
    chn = sample(f(), spl, 100)
    # Check that the log-prior term is calculated in unlinked space.
    @test chn[:logprior] ≈ logpdf.(LogNormal(), chn[:x])
    @test chn[:loglikelihood] ≈ logpdf.(Normal.(chn[:x]), 1.0)
    # This should always be true, but it also indirectly checks that the 
    # log-joint is also calculated in unlinked space.
    @test chn[:lp] ≈ chn[:logprior] + chn[:loglikelihood]
end

end
