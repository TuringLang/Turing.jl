module NumericalTests

using Distributions
using Test: @test, @testset
using Turing: @varname
using HypothesisTests: HypothesisTests

export check_MoGtest_default,
    check_MoGtest_default_z_vector, check_dist_numerical, check_gdemo, check_numerical

# Helper function for numerical tests
function check_numerical(chain, varnames::Vector, exact_vals::Vector; atol=0.2, rtol=0.0)
    for (vn, val) in zip(varnames, exact_vals)
        E = val isa Real ? mean(chain[vn]) : vec(mean(chain[vn]; dims=1))
        @info (varname=vn, exact=val, evaluated=E)
        @test E ≈ val atol = atol rtol = rtol
    end
end

# Wrapper function to quickly check gdemo accuracy.
function check_gdemo(chain; atol=0.2, rtol=0.0)
    return check_numerical(
        chain, [@varname(s), @varname(m)], [49 / 24, 7 / 6]; atol=atol, rtol=rtol
    )
end

# Wrapper function to check MoGtest.
function check_MoGtest_default(chain; atol=0.2, rtol=0.0)
    return check_numerical(
        chain,
        [
            @varname(z1),
            @varname(z2),
            @varname(z3),
            @varname(z4),
            @varname(mu1),
            @varname(mu2)
        ],
        [1.0, 1.0, 2.0, 2.0, 1.0, 4.0];
        atol=atol,
        rtol=rtol,
    )
end

function check_MoGtest_default_z_vector(chain; atol=0.2, rtol=0.0)
    return check_numerical(
        chain,
        [
            @varname(z[1]),
            @varname(z[2]),
            @varname(z[3]),
            @varname(z[4]),
            @varname(mu1),
            @varname(mu2)
        ],
        [1.0, 1.0, 2.0, 2.0, 1.0, 4.0];
        atol=atol,
        rtol=rtol,
    )
end

"""
    two_sample_test(xs_left, xs_right; α=1e-3, warn_on_fail=false)

Perform a two-sample hypothesis test on the two samples `xs_left` and `xs_right`.

Currently the test performed is a Kolmogorov-Smirnov (KS) test.

# Arguments
- `xs_left::AbstractVector`: samples from the first distribution.
- `xs_right::AbstractVector`: samples from the second distribution.

# Keyword arguments
- `α::Real`: significance level for the test. Default: `1e-3`.
- `warn_on_fail::Bool`: whether to warn if the test fails. Default: `false`.
    Makes failures a bit more informative.
"""
function two_sample_test(xs_left, xs_right; α=1e-3, warn_on_fail=false)
    t = HypothesisTests.ApproximateTwoSampleKSTest(xs_left, xs_right)
    # Just a way to make the logs a bit more informative in case of failure.
    if HypothesisTests.pvalue(t) > α
        true
    else
        warn_on_fail &&
            @warn "Two-sample AD test failed with p-value $(HypothesisTests.pvalue(t))"
        warn_on_fail &&
            @warn "Means of the two samples: $(mean(xs_left)), $(mean(xs_right))"
        warn_on_fail &&
            @warn "Variances of the two samples: $(var(xs_left)), $(var(xs_right))"
        false
    end
end

end
