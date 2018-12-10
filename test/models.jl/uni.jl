using Test, Turing
turnprogress(false)

n_samples = 20_000
mean_atol = 0.25
var_atol = 1.0

for dist ∈ [Normal(0, 1), Gamma(2, 3), InverseGamma(2, 3), Beta(2, 1), Cauchy(0, 1)]

    @testset "$(string(typeof(dist)))" begin

        function mf(vi::Turing.VarInfo, sampler::Turing.AnySampler; )::Any
            vi.logp = zero(Real)
            varname = Turing.VarName(:mf, :x, "", 1)
            x, _lp = Turing.assume(sampler, dist, varname, vi)
            vi.logp += _lp
        end

        chn = sample(mf, NUTS(n_samples, 0.8))
        chn_xs = chn[:x][1:2:end] # thining by halving

        # Mean
        dist_mean = mean(dist)
        if ~isnan(dist_mean) && ~isinf(dist_mean)
            chn_mean = mean(chn_xs)
            @test chn_mean ≈ dist_mean atol=mean_atol
        end

        # Variance
        dist_var = var(dist)
        if ~isnan(dist_var) && ~isinf(dist_var)
            chn_var = var(chn_xs)
            @test chn_var ≈ dist_var atol=var_atol
        end

    end

end
