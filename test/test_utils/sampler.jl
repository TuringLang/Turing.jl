module SamplerTestUtils

using Random
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

"""
Check that sampling is deterministic when using the same RNG seed.
"""
function test_rng_respected(spl)
    @model function f(z)
        # put at least two variables here so that we can meaningfully test Gibbs
        x ~ Normal()
        y ~ Normal()
        return z ~ Normal(x + y)
    end
    model = f(2.0)
    chn1 = sample(Xoshiro(468), model, spl, 100)
    chn2 = sample(Xoshiro(468), model, spl, 100)
    @test isapprox(chn1[:x], chn2[:x])
    @test isapprox(chn1[:y], chn2[:y])
end

end
