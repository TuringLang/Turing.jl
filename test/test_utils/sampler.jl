module SamplerTestUtils

using AbstractMCMC
using AbstractPPL
using DynamicPPL
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
    @test chn[:logprior] ≈ logpdf.(LogNormal(), chn[@varname(x)])
    @test chn[:loglikelihood] ≈ logpdf.(Normal.(chn[@varname(x)]), 1.0)
    # This should always be true, but it also indirectly checks that the 
    # log-joint is also calculated in unlinked space.
    @test chn[:logjoint] ≈ chn[:logprior] + chn[:loglikelihood]
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

"""
    test_sampler_analytical(models, sampler, args...; kwargs...)

Test that `sampler` produces correct marginal posterior means on each model in `models`.

In short, this method iterates through `models`, calls `AbstractMCMC.sample` on the `model`
and `sampler` to produce a `chain`, and then checks the chain's mean for every (leaf)
varname `vn` against the corresponding value returned by
`DynamicPPL.TestUtils.posterior_mean` for each model.

For this to work, each model in `models` must have a known analytical posterior mean
that can be computed by `DynamicPPL.TestUtils.posterior_mean`.

# Arguments
- `models`: A collection of instances of `DynamicPPL.Model` to test on.
- `sampler`: The `AbstractMCMC.AbstractSampler` to test.
- `args...`: Arguments forwarded to `sample`.

# Keyword arguments
- `varnames_filter`: A filter to apply to `varnames(model)`, allowing comparison for only
    a subset of the varnames.
- `atol=1e-1`: Absolute tolerance used in `@test`.
- `rtol=1e-3`: Relative tolerance used in `@test`.
- `kwargs...`: Keyword arguments forwarded to `sample`.
"""
function test_sampler_analytical(
    models,
    sampler::AbstractMCMC.AbstractSampler,
    args...;
    varnames_filter=Returns(true),
    atol=1e-1,
    rtol=1e-3,
    sampler_name=typeof(sampler),
    kwargs...,
)
    @testset "$(sampler_name) on $(nameof(model))" for model in models
        chain = AbstractMCMC.sample(model, sampler, args...; kwargs...)
        target_values = DynamicPPL.TestUtils.posterior_mean(model)
        for vn in filter(varnames_filter, DynamicPPL.TestUtils.varnames(model))
            # We want to compare elementwise which can be achieved by
            # extracting the leaves of the `VarName` and the corresponding value.
            for vn_leaf in AbstractPPL.varname_leaves(vn, get(target_values, vn))
                target_value = get(target_values, vn_leaf)
                chain_mean_value = mean(chain[vn_leaf])
                @test chain_mean_value ≈ target_value atol = atol rtol = rtol
            end
        end
    end
end

end
