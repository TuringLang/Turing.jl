module NumericalTests

using Distributions
using MCMCChains: namesingroup
using Test: @test, @testset

export check_MoGtest_default, check_MoGtest_default_z_vector, check_dist_numerical,
    check_gdemo, check_numerical

function check_dist_numerical(dist, chn; mean_tol = 0.1, var_atol = 1.0, var_tol = 0.5)
    @testset "numerical" begin
        # Extract values.
        chn_xs = Array(chn[1:2:end, namesingroup(chn, :x), :])

        # Check means.
        dist_mean = mean(dist)
        mean_shape = size(dist_mean)
        if !all(isnan, dist_mean) && !all(isinf, dist_mean)
            chn_mean = vec(mean(chn_xs, dims=1))
            chn_mean = length(chn_mean) == 1 ?
                chn_mean[1] :
                reshape(chn_mean, mean_shape)
            atol_m = length(chn_mean) > 1 ?
                mean_tol * length(chn_mean) :
                max(mean_tol, mean_tol * chn_mean)
            @test chn_mean ≈ dist_mean atol=atol_m
        end

        # Check variances.
        # var() for Distributions.MatrixDistribution is not defined
        if !(dist isa Distributions.MatrixDistribution)
            # Variance
            dist_var = var(dist)
            var_shape = size(dist_var)
            if !all(isnan, dist_var) && !all(isinf, dist_var)
                chn_var = vec(var(chn_xs, dims=1))
                chn_var = length(chn_var) == 1 ?
                    chn_var[1] :
                    reshape(chn_var, var_shape)
                atol_v = length(chn_mean) > 1 ?
                    mean_tol * length(chn_mean) :
                    max(mean_tol, mean_tol * chn_mean)
                @test chn_mean ≈ dist_mean atol=atol_v
            end
        end
    end
end

# Helper function for numerical tests
function check_numerical(chain,
                        symbols::Vector,
                        exact_vals::Vector;
                        atol=0.2,
                        rtol=0.0)
    for (sym, val) in zip(symbols, exact_vals)
        E = val isa Real ?
            mean(chain[sym]) :
            vec(mean(chain[sym], dims=1))
        @info (symbol=sym, exact=val, evaluated=E)
        @test E ≈ val atol=atol rtol=rtol
    end
end

# Wrapper function to quickly check gdemo accuracy.
function check_gdemo(chain; atol=0.2, rtol=0.0)
    check_numerical(chain, [:s, :m], [49/24, 7/6], atol=atol, rtol=rtol)
end

# Wrapper function to check MoGtest.
function check_MoGtest_default(chain; atol=0.2, rtol=0.0)
    check_numerical(chain,
        [:z1, :z2, :z3, :z4, :mu1, :mu2],
        [1.0, 1.0, 2.0, 2.0, 1.0, 4.0],
        atol=atol, rtol=rtol)
end

function check_MoGtest_default_z_vector(chain; atol=0.2, rtol=0.0)
    check_numerical(chain,
        [Symbol("z[1]"), Symbol("z[2]"), Symbol("z[3]"), Symbol("z[4]"), :mu1, :mu2],
        [1.0, 1.0, 2.0, 2.0, 1.0, 4.0],
        atol=atol, rtol=rtol)
end

"""
    two_sample_ad_test(xs_left, xs_right; α=1e-3)

Perform a two-sample Anderson-Darling (AD) test on the two samples `xs_left` and `xs_right`.
"""
function two_sample_ad_test(xs_left, xs_right; α=1e-3)
    t = KSampleADTest(xs_left, xs_right)
    return pvalue(t) > α
end

end
