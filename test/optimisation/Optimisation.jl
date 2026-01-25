module OptimisationTests

using AbstractPPL: AbstractPPL
using Bijectors: Bijectors
import DifferentiationInterface as DI
using Distributions
using DynamicPPL: DynamicPPL
using ForwardDiff: ForwardDiff
using LinearAlgebra: Diagonal, I
using Random: Random
using Optimization
using Optimization: Optimization
using OptimizationBBO: OptimizationBBO
using OptimizationNLopt: OptimizationNLopt
using OptimizationOptimJL: OptimizationOptimJL
using Random: Random
using ReverseDiff: ReverseDiff
using StableRNGs: StableRNG
using StatsBase: StatsBase
using StatsBase: coef, coefnames, coeftable, informationmatrix, stderror, vcov
using Test: @test, @testset, @test_throws
using Turing
using Turing.Optimisation:
    ModeResult, InitWithConstraintCheck, satisfies_constraints, make_optim_bounds_and_init

SECOND_ORDER_ADTYPE = DI.SecondOrder(AutoForwardDiff(), AutoForwardDiff())
GDEMO_DEFAULT = DynamicPPL.TestUtils.demo_assume_observe_literal()

function check_optimisation_result(
    result::ModeResult,
    true_values::AbstractDict{<:AbstractPPL.VarName,<:Any},
    true_logp::Real,
    check_retcode=true,
)
    # Check that `result.params` contains all the keys in `true_values`
    @test Set(keys(result.params)) == Set(keys(true_values))
    # Check that their values are close
    for (vn, val) in result.params
        @test isapprox(val, true_values[vn], atol=0.01)
    end
    # Check logp and retcode
    @test isapprox(result.lp, true_logp, atol=0.01)
    if check_retcode
        @test result.optim_result.retcode == Optimization.ReturnCode.Success
    end
end

@testset "Initialisation" begin
    @testset "satisfies_constraints" begin
        @testset "univariate" begin
            val = 0.0
            dist = Normal() # only used for dispatch
            @test satisfies_constraints(nothing, nothing, val, dist)
            @test satisfies_constraints(-1.0, nothing, val, dist)
            @test !satisfies_constraints(1.0, nothing, val, dist)
            @test satisfies_constraints(nothing, 1.0, val, dist)
            @test !satisfies_constraints(nothing, -1.0, val, dist)
            @test satisfies_constraints(-1.0, 1.0, val, dist)
        end

        @testset "univariate ForwardDiff.Dual" begin
            val = ForwardDiff.Dual(0.0, 1.0)
            dist = Normal() # only used for dispatch
            @test satisfies_constraints(nothing, 0.0, val, dist)
            @test !satisfies_constraints(nothing, -0.01, val, dist)
            val = ForwardDiff.Dual(0.0, -1.0)
            @test satisfies_constraints(0.0, nothing, val, dist)
            @test !satisfies_constraints(0.01, nothing, val, dist)
        end

        @testset "multivariate" begin
            val = [0.3, 0.5, 0.2]
            dist = Dirichlet(ones(3)) # only used for dispatch
            @test satisfies_constraints(nothing, nothing, val, dist)
            @test satisfies_constraints(zeros(3), nothing, val, dist)
            @test !satisfies_constraints(ones(3), nothing, val, dist)
            @test satisfies_constraints(nothing, ones(3), val, dist)
            @test !satisfies_constraints(nothing, zeros(3), val, dist)
            @test satisfies_constraints(zeros(3), ones(3), val, dist)
            @test !satisfies_constraints([0.4, 0.0, 0.0], nothing, val, dist)
            @test !satisfies_constraints(nothing, [1.0, 1.0, 0.1], val, dist)
        end

        @testset "multivariate ForwardDiff.Dual" begin
            val = [ForwardDiff.Dual(0.5, 1.0), ForwardDiff.Dual(0.5, -1.0)]
            dist = Dirichlet(ones(3)) # only used for dispatch
            @test satisfies_constraints([0.5, 0.5], [0.5, 0.5], val, dist)
        end

        @testset "Matrix distributions" begin
            dist = Wishart(3, [0.5 0.0; 0.0 0.5]) # only used for dispatch
            val = [1.0 0.0; 0.0 1.0]
            @test satisfies_constraints(zeros(2, 2), ones(2, 2), val, dist)
            @test satisfies_constraints(nothing, ones(2, 2), val, dist)
            @test satisfies_constraints(zeros(2, 2), nothing, val, dist)
            val = [2.0 -1.0; -1.0 2.0]
            @test !satisfies_constraints(zeros(2, 2), ones(2, 2), val, dist)
            @test !satisfies_constraints(nothing, ones(2, 2), val, dist)
            @test !satisfies_constraints(zeros(2, 2), nothing, val, dist)
        end

        @testset "LKJCholesky" begin
            dist = LKJCholesky(3, 0.5)
            val = rand(dist)
            @test satisfies_constraints(nothing, nothing, val, dist)
            # Just refuse to handle these.
            @test_throws ArgumentError satisfies_constraints(
                zeros(3, 3), nothing, val, dist
            )
            @test_throws ArgumentError satisfies_constraints(nothing, ones(3, 3), val, dist)
        end
    end

    @testset "errors when constraints can't be satisfied" begin
        @model function diric()
            x ~ Dirichlet(ones(2))
            return 1.0 ~ Normal()
        end
        ldf = LogDensityFunction(diric())
        # These are all impossible constraints for a Dirichlet(ones(2))
        for (lb, ub) in
            [([2.0, 2.0], nothing), (nothing, [-1.0, -1.0]), ([0.3, 0.3], [0.1, 0.1])]
            # unit test the function
            @test_throws ArgumentError make_optim_bounds_and_init(
                Random.default_rng(), ldf, InitFromPrior(), (x=lb,), (x=ub,)
            )
            # check that the high-level function also errors
            @test_throws ArgumentError maximum_likelihood(diric(); lb=(x=lb,), ub=(x=ub,))
            @test_throws ArgumentError maximum_a_posteriori(diric(); lb=(x=lb,), ub=(x=ub,))
        end

        # Try to provide reasonable constraints, but bad initial params
        @model function normal_model()
            x ~ Normal()
            return 1.0 ~ Normal(x)
        end
        ldf = LogDensityFunction(normal_model())
        lb = (x=-1.0,)
        ub = (x=1.0,)
        bad_init = (x=10.0,)
        @test_throws ArgumentError make_optim_bounds_and_init(
            Random.default_rng(), ldf, InitFromParams(bad_init), lb, ub
        )
        @test_throws ArgumentError maximum_likelihood(
            normal_model(); initial_params=InitFromParams(bad_init), lb=lb, ub=ub
        )
        @test_throws ArgumentError maximum_a_posteriori(
            normal_model(); initial_params=InitFromParams(bad_init), lb=lb, ub=ub
        )
    end

    @testset "generation of vector constraints" begin
        @testset "$dist" for (lb, ub, dist) in (
            ((x=0.1,), (x=0.5,), Normal()),
            ((x=0.1,), (x=0.5,), Beta(2, 2)),
            ((x=[0.1, 0.1],), (x=[0.5, 0.5],), MvNormal(zeros(2), I)),
            (
                (x=[0.1, 0.1],),
                (x=[0.5, 0.5],),
                product_distribution([Beta(2, 2), Beta(2, 2)]),
            ),
            # TODO(penelopeysm): Broken because DynamicPPL.to_vec_transform(dist) fails.
            # This needs an upstream fix.
            # ((x=(a=0.1, b=0.1),), (x=(a=0.5, b=0.5),), product_distribution((a=Beta(2, 2), b=Beta(2, 2)))),
        )
            @model f() = x ~ dist
            model = f()
            maybe_to_vec(x::AbstractVector) = x
            maybe_to_vec(x) = [x]

            @testset "unlinked" begin
                ldf = LogDensityFunction(model)
                lb_vec, ub_vec, init_vec = make_optim_bounds_and_init(
                    Random.default_rng(), ldf, InitFromPrior(), lb, ub
                )
                @test lb_vec == maybe_to_vec(lb.x)
                @test ub_vec == maybe_to_vec(ub.x)
                @test all(init_vec .>= lb_vec)
                @test all(init_vec .<= ub_vec)
            end

            @testset "linked" begin
                vi = DynamicPPL.link!!(DynamicPPL.VarInfo(model), model)
                ldf = LogDensityFunction(model, DynamicPPL.getlogjoint, vi)
                lb_vec, ub_vec, init_vec = make_optim_bounds_and_init(
                    Random.default_rng(), ldf, InitFromPrior(), lb, ub
                )
                b = Bijectors.bijector(dist)
                @test lb_vec ≈ maybe_to_vec(b(lb.x))
                @test ub_vec ≈ maybe_to_vec(b(ub.x))
                @test all(init_vec .>= lb_vec)
                @test all(init_vec .<= ub_vec)
            end
        end
    end

    @testset "forbidding linked + constraints for complicated distributions" begin
        @testset for dist in (LKJCholesky(3, 1.0), Dirichlet(ones(3)))
            @model f() = x ~ dist
            model = f()

            vi = DynamicPPL.link!!(DynamicPPL.VarInfo(model), model)
            ldf = LogDensityFunction(model, DynamicPPL.getlogjoint, vi)
            lb = (x=rand(dist),)
            ub = (;)

            @test_throws ArgumentError make_optim_bounds_and_init(
                Random.default_rng(), ldf, InitFromPrior(), lb, ub
            )
            @test_throws ArgumentError maximum_likelihood(model; lb=lb, ub=ub, link=true)
            @test_throws ArgumentError maximum_a_posteriori(model; lb=lb, ub=ub, link=true)
        end
    end
end

@testset "Optimisation" begin
    # The `stats` field is populated only in newer versions of OptimizationOptimJL and
    # similar packages. Hence we end up doing this check a lot
    hasstats(result) = result.optim_result.stats !== nothing

    @testset "errors on invalid model" begin
        @model function invalid_model()
            x ~ Normal()
            return x ~ Beta()
        end
        m = invalid_model()
        @test_throws ErrorException maximum_likelihood(m)
        @test_throws ErrorException maximum_a_posteriori(m)
    end

    @testset "gdemo" begin
        @testset "MLE" begin
            true_value = Dict(@varname(s) => 0.0625, @varname(m) => 1.75)
            true_logp = loglikelihood(GDEMO_DEFAULT, true_value)
            check_success(result) = check_optimisation_result(result, true_value, true_logp)

            m1 = Turing.Optimisation.estimate_mode(GDEMO_DEFAULT, MLE())
            m2 = maximum_likelihood(
                StableRNG(468),
                GDEMO_DEFAULT,
                OptimizationOptimJL.LBFGS();
                initial_params=InitFromParams(true_value),
            )
            m3 = maximum_likelihood(
                StableRNG(468),
                GDEMO_DEFAULT,
                OptimizationOptimJL.Newton();
                adtype=SECOND_ORDER_ADTYPE,
            )
            m4 = maximum_likelihood(
                StableRNG(468),
                GDEMO_DEFAULT,
                OptimizationOptimJL.BFGS();
                adtype=AutoReverseDiff(),
            )
            m5 = maximum_likelihood(
                StableRNG(468),
                GDEMO_DEFAULT,
                OptimizationOptimJL.NelderMead();
                initial_params=InitFromParams(true_value),
            )
            m6 = maximum_likelihood(
                StableRNG(468), GDEMO_DEFAULT, OptimizationOptimJL.NelderMead()
            )

            check_success(m1)
            check_success(m2)
            check_success(m3)
            check_success(m4)
            check_success(m5)
            check_success(m6)

            @test !hasstats(m2) || m2.optim_result.stats.iterations <= 1
            if hasstats(m6) && hasstats(m5)
                @test m5.optim_result.stats.iterations < m6.optim_result.stats.iterations
            end

            @test !hasstats(m2) || m2.optim_result.stats.gevals > 0
            @test !hasstats(m3) || m3.optim_result.stats.gevals > 0
            @test !hasstats(m4) || m4.optim_result.stats.gevals > 0
            @test !hasstats(m5) || m5.optim_result.stats.gevals == 0
            @test !hasstats(m6) || m6.optim_result.stats.gevals == 0
        end

        @testset "MAP" begin
            true_value = Dict(@varname(s) => 49 / 54, @varname(m) => 7 / 6)
            true_logp = logjoint(GDEMO_DEFAULT, true_value)
            check_success(result) = check_optimisation_result(result, true_value, true_logp)

            m1 = Turing.Optimisation.estimate_mode(StableRNG(468), GDEMO_DEFAULT, MAP())
            m2 = maximum_a_posteriori(
                StableRNG(468),
                GDEMO_DEFAULT,
                OptimizationOptimJL.LBFGS();
                initial_params=InitFromParams(true_value),
            )
            m3 = maximum_a_posteriori(
                StableRNG(468),
                GDEMO_DEFAULT,
                OptimizationOptimJL.Newton();
                adtype=SECOND_ORDER_ADTYPE,
            )
            m4 = maximum_a_posteriori(
                StableRNG(468),
                GDEMO_DEFAULT,
                OptimizationOptimJL.BFGS();
                adtype=AutoReverseDiff(),
            )
            m5 = maximum_a_posteriori(
                StableRNG(468),
                GDEMO_DEFAULT,
                OptimizationOptimJL.NelderMead();
                initial_params=InitFromParams(true_value),
            )
            m6 = maximum_a_posteriori(
                StableRNG(468), GDEMO_DEFAULT, OptimizationOptimJL.NelderMead()
            )

            check_success(m1)
            check_success(m2)
            check_success(m3)
            check_success(m4)
            check_success(m5)
            check_success(m6)

            @test !hasstats(m2) || m2.optim_result.stats.iterations <= 1
            if hasstats(m6) && hasstats(m5)
                @test m5.optim_result.stats.iterations < m6.optim_result.stats.iterations
            end

            @test !hasstats(m2) || m2.optim_result.stats.gevals > 0
            @test !hasstats(m3) || m3.optim_result.stats.gevals > 0
            @test !hasstats(m4) || m4.optim_result.stats.gevals > 0
            @test !hasstats(m5) || m5.optim_result.stats.gevals == 0
            @test !hasstats(m6) || m6.optim_result.stats.gevals == 0
        end

        @testset "MLE with box constraints" begin
            true_value = Dict(@varname(s) => 0.0625, @varname(m) => 1.75)
            true_logp = loglikelihood(GDEMO_DEFAULT, true_value)
            check_success(result) = check_optimisation_result(result, true_value, true_logp)

            lb = (s=0.0, m=0.0)
            ub = (s=2.0, m=2.0)
            # We need to disable linking during the optimisation here, because it will
            # result in NaN's. See the comment on allowed_incorrect_mle below. In fact
            # even sometimes without linking it still gets NaN's -- we get round that
            # in these tests by seeding the RNG.
            kwargs = (; lb=lb, ub=ub, link=false)

            m1 = Turing.Optimisation.estimate_mode(
                StableRNG(468), GDEMO_DEFAULT, MLE(); kwargs...
            )
            m2 = maximum_likelihood(
                StableRNG(468),
                GDEMO_DEFAULT,
                OptimizationOptimJL.Fminbox(OptimizationOptimJL.LBFGS());
                initial_params=InitFromParams(true_value),
                kwargs...,
            )
            m3 = maximum_likelihood(
                StableRNG(468),
                GDEMO_DEFAULT,
                OptimizationBBO.BBO_separable_nes();
                maxiters=100_000,
                abstol=1e-5,
                kwargs...,
            )
            m4 = maximum_likelihood(
                StableRNG(468),
                GDEMO_DEFAULT,
                OptimizationOptimJL.Fminbox(OptimizationOptimJL.BFGS());
                adtype=AutoReverseDiff(),
                kwargs...,
            )
            m5 = maximum_likelihood(
                StableRNG(468),
                GDEMO_DEFAULT,
                OptimizationOptimJL.IPNewton();
                initial_params=InitFromParams(true_value),
                adtype=SECOND_ORDER_ADTYPE,
                kwargs...,
            )
            m6 = maximum_likelihood(StableRNG(468), GDEMO_DEFAULT; kwargs...)

            check_success(m1)
            check_success(m2)
            check_success(m3)
            check_success(m4)
            check_success(m5)
            check_success(m6)

            @test !hasstats(m2) || m2.optim_result.stats.iterations <= 1
            @test !hasstats(m5) || m5.optim_result.stats.iterations <= 1

            @test !hasstats(m2) || m2.optim_result.stats.gevals > 0
            @test !hasstats(m3) || m3.optim_result.stats.gevals == 0
            @test !hasstats(m4) || m4.optim_result.stats.gevals > 0
            @test !hasstats(m5) || m5.optim_result.stats.gevals > 0
        end

        @testset "MAP with box constraints" begin
            true_value = Dict(@varname(s) => 49 / 54, @varname(m) => 7 / 6)
            true_logp = logjoint(GDEMO_DEFAULT, true_value)
            check_success(result) = check_optimisation_result(result, true_value, true_logp)

            lb = (s=0.0, m=0.0)
            ub = (s=2.0, m=2.0)
            # We need to disable linking during the optimisation here, because it will
            # result in NaN's. See the comment on allowed_incorrect_mle below.
            kwargs = (; lb=lb, ub=ub, link=false)

            m1 = Turing.Optimisation.estimate_mode(
                StableRNG(468), GDEMO_DEFAULT, MAP(); kwargs...
            )
            m2 = maximum_a_posteriori(
                StableRNG(468),
                GDEMO_DEFAULT,
                OptimizationOptimJL.Fminbox(OptimizationOptimJL.LBFGS());
                initial_params=InitFromParams(true_value),
                kwargs...,
            )
            m3 = maximum_a_posteriori(
                StableRNG(468),
                GDEMO_DEFAULT,
                OptimizationBBO.BBO_separable_nes();
                maxiters=100_000,
                abstol=1e-5,
                kwargs...,
            )
            m4 = maximum_a_posteriori(
                StableRNG(468),
                GDEMO_DEFAULT,
                OptimizationOptimJL.Fminbox(OptimizationOptimJL.BFGS());
                adtype=AutoReverseDiff(),
                kwargs...,
            )
            m5 = maximum_a_posteriori(
                StableRNG(468),
                GDEMO_DEFAULT,
                OptimizationOptimJL.IPNewton();
                initial_params=InitFromParams(true_value),
                adtype=SECOND_ORDER_ADTYPE,
                kwargs...,
            )
            m6 = maximum_a_posteriori(StableRNG(468), GDEMO_DEFAULT; kwargs...)

            check_success(m1)
            check_success(m2)
            check_success(m3)
            check_success(m4)
            check_success(m5)
            check_success(m6)

            @test !hasstats(m2) || m2.optim_result.stats.iterations <= 1
            @test !hasstats(m5) || m5.optim_result.stats.iterations <= 1
            @test !hasstats(m2) || m2.optim_result.stats.gevals > 0
            @test !hasstats(m3) || m3.optim_result.stats.gevals == 0
            @test !hasstats(m4) || m4.optim_result.stats.gevals > 0
            @test !hasstats(m5) || m5.optim_result.stats.gevals > 0
        end
    end

    @testset "StatsBase integration" begin
        true_s = 0.0625
        true_m = 1.75
        true_value = Dict(@varname(s) => true_s, @varname(m) => true_m)
        true_lp = loglikelihood(GDEMO_DEFAULT, true_value)
        mle_est = maximum_likelihood(GDEMO_DEFAULT)

        @test coefnames(mle_est) == [@varname(s), @varname(m)]
        @test coefnames(mle_est) == params(mle_est)

        diffs = coef(mle_est) .- [true_s, true_m]
        @test all(isapprox.(diffs, 0.0, atol=0.1))

        infomat = [2/(2 * true_s^2) 0.0; 0.0 2/true_s]
        @test all(isapprox.(infomat - informationmatrix(mle_est), 0.0, atol=0.01))

        @test vcov(mle_est) == inv(informationmatrix(mle_est))
        vcovmat = [2 * true_s^2/2 0.0; 0.0 true_s/2]
        @test all(isapprox.(vcovmat - vcov(mle_est), 0.0, atol=0.01))

        ctable = coeftable(mle_est)
        @test ctable isa StatsBase.CoefTable

        s = stderror(mle_est)
        @test all(isapprox.(s - [0.06250415643292194, 0.17677963626053916], 0.0, atol=0.01))

        @test isapprox(loglikelihood(mle_est), true_lp, atol=0.01)
    end

    @testset "Linear regression test" begin
        @model function regtest(x, y)
            beta ~ MvNormal(zeros(2), I)
            mu = x * beta
            return y ~ MvNormal(mu, I)
        end

        true_beta = [1.0, -2.2]
        x = rand(StableRNG(468), 40, 2)
        y = x * true_beta

        model = regtest(x, y)
        mle = maximum_likelihood(StableRNG(468), model)

        vcmat = inv(x'x)
        vcmat_mle = vcov(mle)

        @test isapprox(mle.params[@varname(beta)], true_beta)
        @test isapprox(vcmat, vcmat_mle)
    end

    @testset "Dot tilde test" begin
        @model function dot_gdemo(x)
            s ~ InverseGamma(2, 3)
            m ~ Normal(0, sqrt(s))

            return (.~)(x, Normal(m, sqrt(s)))
        end

        model_dot = dot_gdemo([1.5, 2.0])

        mle1 = maximum_likelihood(GDEMO_DEFAULT)
        mle2 = maximum_likelihood(model_dot)

        map1 = maximum_a_posteriori(GDEMO_DEFAULT)
        map2 = maximum_a_posteriori(model_dot)

        @test isapprox(mle1.params[@varname(s)], mle2.params[@varname(s)])
        @test isapprox(mle1.params[@varname(m)], mle2.params[@varname(m)])
        @test isapprox(map1.params[@varname(s)], map2.params[@varname(s)])
        @test isapprox(map1.params[@varname(m)], map2.params[@varname(m)])
    end

    @testset "MAP for $(model.f)" for model in DynamicPPL.TestUtils.DEMO_MODELS
        true_optima = DynamicPPL.TestUtils.posterior_optima(model)

        optimizers = [
            (false, OptimizationOptimJL.LBFGS()),
            (false, OptimizationOptimJL.NelderMead()),
            (true, OptimizationNLopt.NLopt.LD_TNEWTON_PRECOND_RESTART()),
        ]
        @testset "$(nameof(typeof(optimizer)))" for (needs_second_order, optimizer) in
                                                    optimizers
            adtype = if needs_second_order
                SECOND_ORDER_ADTYPE
            else
                AutoForwardDiff()
            end
            result = maximum_a_posteriori(StableRNG(468), model, optimizer; adtype=adtype)

            for vn in DynamicPPL.TestUtils.varnames(model)
                val = AbstractPPL.getvalue(true_optima, vn)
                for vn_leaf in AbstractPPL.varname_leaves(vn, val)
                    expected = AbstractPPL.getvalue(true_optima, vn_leaf)
                    actual = result.params[vn_leaf]
                    @test expected ≈ actual atol = 0.05
                end
            end
        end
    end

    # Some of the models have one variance parameter per observation, and so
    # the MLE should have the variances set to 0. Since we're working in
    # transformed space, this corresponds to `-Inf`, which is of course not achievable.
    # In particular, it can result in "early termination" of the optimization process
    # because we hit NaNs, etc. To avoid this, we set the `g_tol` and the `f_tol` to
    # something larger than the default.
    allowed_incorrect_mle = [
        DynamicPPL.TestUtils.demo_dot_assume_observe,
        DynamicPPL.TestUtils.demo_assume_index_observe,
        DynamicPPL.TestUtils.demo_assume_multivariate_observe,
        DynamicPPL.TestUtils.demo_assume_multivariate_observe_literal,
        DynamicPPL.TestUtils.demo_dot_assume_observe_submodel,
        DynamicPPL.TestUtils.demo_dot_assume_observe_matrix_index,
        DynamicPPL.TestUtils.demo_assume_submodel_observe_index_literal,
        DynamicPPL.TestUtils.demo_dot_assume_observe_index,
        DynamicPPL.TestUtils.demo_dot_assume_observe_index_literal,
        DynamicPPL.TestUtils.demo_assume_matrix_observe_matrix_index,
    ]
    @testset "MLE for $(model.f)" for model in DynamicPPL.TestUtils.DEMO_MODELS
        true_optima = DynamicPPL.TestUtils.likelihood_optima(model)

        optimizers = [
            (false, OptimizationOptimJL.LBFGS()),
            (false, OptimizationOptimJL.NelderMead()),
            (true, OptimizationNLopt.NLopt.LD_TNEWTON_PRECOND_RESTART()),
        ]
        @testset "$(nameof(typeof(optimizer)))" for (needs_second_order, optimizer) in
                                                    optimizers
            try
                adtype = if needs_second_order
                    SECOND_ORDER_ADTYPE
                else
                    AutoForwardDiff()
                end
                result = maximum_likelihood(model, optimizer; reltol=1e-3, adtype=adtype)

                for vn in DynamicPPL.TestUtils.varnames(model)
                    val = AbstractPPL.getvalue(true_optima, vn)
                    for vn_leaf in AbstractPPL.varname_leaves(vn, val)
                        expected = AbstractPPL.getvalue(true_optima, vn_leaf)
                        actual = result.params[vn_leaf]
                        if model.f in allowed_incorrect_mle
                            @test isfinite(actual)
                        else
                            @test expected ≈ actual atol = 0.05
                        end
                    end
                end
            catch e
                if model.f in allowed_incorrect_mle
                    @info "MLE test for $(model.f) errored, but this is expected due to variance MLE being zero"
                else
                    rethrow(e)
                end
            end
        end
    end

    @testset "distribution with dynamic support and constraints" begin
        @model function f()
            x ~ Uniform(-5, 5)
            return y ~ truncated(Normal(); lower=x)
        end
        # Note that in this testset we don't need to seed RNG because the initial params
        # are fully specified.
        inits = (x=0.0, y=2.5)
        lb = (y=2.0,)
        # Here, the constraints that are passed to Optimization.jl are not fully 'correct',
        # because the constraints were determined with a single static value of `x`. Thus,
        # during the optimization it is possible for `y` to go out of bounds. We check that
        # such cases are caught.
        @test_throws DomainError maximum_a_posteriori(
            f(); initial_params=InitFromParams(inits), lb=lb, link=true
        )
        # If there is no linking, then the constraints are no longer incorrect (they are
        # always static). So the optimization should succeed (it might give silly results,
        # but that's not our problem).
        @test maximum_a_posteriori(
            f(); initial_params=InitFromParams(inits), lb=lb, link=false
        ) isa ModeResult
        # If the user wants to disable it, they should be able to.
        @test maximum_a_posteriori(
            f();
            initial_params=InitFromParams(inits),
            lb=lb,
            link=true,
            check_constraints_at_runtime=false,
        ) isa ModeResult
    end

    @testset "using ModeResult to initialise MCMC" begin
        @model function f(y)
            μ ~ Normal(0, 1)
            σ ~ Gamma(2, 1)
            return y ~ Normal(μ, σ)
        end
        model = f(randn(10))
        mle = maximum_likelihood(model)
        # TODO(penelopeysm): This relies on the fact that HMC does indeed
        # use the initial_params passed to it. We should use something
        # like a StaticSampler (see test/mcmc/Inference) to make this more
        # robust.
        chain = sample(
            model, HMC(0.1, 10), 2; initial_params=InitFromParams(mle), num_warmup=0
        )
        # Check that those parameters were indeed used as initial params
        @test chain[:µ][1] == mle.params[@varname(µ)]
        @test chain[:σ][1] == mle.params[@varname(σ)]
    end

    @testset "returned on ModeResult" begin
        @model function f()
            x ~ Normal()
            2.0 ~ Normal(x)
            return x + 1.0
        end
        model = f()
        result = maximum_a_posteriori(model)
        @test returned(model, result) == result.params[@varname(x)] + 1.0
        result = maximum_likelihood(model)
        @test returned(model, result) == result.params[@varname(x)] + 1.0
    end

    # Issue: https://discourse.julialang.org/t/turing-mixture-models-with-dirichlet-weightings/112910
    @testset "Optimization with different linked dimensionality" begin
        @model demo_dirichlet() = x ~ Dirichlet(2 * ones(3))
        model = demo_dirichlet()
        result = maximum_a_posteriori(model)
        @test result.params[@varname(x)] ≈ mode(Dirichlet(2 * ones(3))) atol = 0.2
    end

    @testset "with :=" begin
        @model function demo_track()
            x ~ Normal()
            return y := 100 + x
        end
        model = demo_track()
        result = maximum_a_posteriori(model)
        @test result.params[@varname(x)] ≈ 0 atol = 1e-1
        @test result.params[@varname(y)] ≈ 100 atol = 1e-1
    end

    @testset "Collinear coeftable" begin
        xs = [-1.0, 0.0, 1.0]
        ys = [0.0, 0.0, 0.0]

        @model function collinear(x, y)
            a ~ Normal(0, 1)
            b ~ Normal(0, 1)
            return y ~ MvNormal(a .* x .+ b .* x, 1)
        end

        model = collinear(xs, ys)
        mle_estimate = maximum_likelihood(model)
        tab = coeftable(mle_estimate)
        @test isnan(tab.cols[2][1])
        @test tab.colnms[end] == "Error notes"
        @test occursin("singular", tab.cols[end][1])
    end

    @testset "Negative variance" begin
        # A model for which the likelihood has a saddle point at x=0, y=0.
        # Creating an optimisation result for this model at the x=0, y=0 results in negative
        # variance for one of the variables, because the variance is calculated as the
        # diagonal of the inverse of the Hessian.
        @model function saddle_model()
            x ~ Normal(0, 1)
            y ~ Normal(x, 1)
            @addlogprob! x^2 - y^2
            return nothing
        end
        m = Turing.Optimisation.ModeResult(
            MLE(),
            DynamicPPL.VarNamedTuple((; x=0.0, y=0.0)),
            0.0,
            false,
            DynamicPPL.LogDensityFunction(saddle_model(), DynamicPPL.getloglikelihood),
            nothing,
        )
        ct = coeftable(m)
        @test isnan(ct.cols[2][1])
        @test ct.colnms[end] == "Error notes"
        @test occursin("Negative variance", ct.cols[end][1])
    end

    @testset "Same coeftable with/without numerrors_warnonly" begin
        xs = [0.0, 1.0, 2.0]

        @model function extranormal(x)
            mean ~ Normal(0, 1)
            return x ~ Normal(mean, 1)
        end

        model = extranormal(xs)
        mle_estimate = maximum_likelihood(model)
        warnonly_coeftable = coeftable(mle_estimate; numerrors_warnonly=true)
        no_warnonly_coeftable = coeftable(mle_estimate; numerrors_warnonly=false)
        @test warnonly_coeftable.cols == no_warnonly_coeftable.cols
        @test warnonly_coeftable.colnms == no_warnonly_coeftable.colnms
        @test warnonly_coeftable.rownms == no_warnonly_coeftable.rownms
        @test warnonly_coeftable.pvalcol == no_warnonly_coeftable.pvalcol
        @test warnonly_coeftable.teststatcol == no_warnonly_coeftable.teststatcol
    end
end

end
