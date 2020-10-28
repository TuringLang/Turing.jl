using Turing
using Optim
using BlackBoxOptim
using GalacticOptim
using Test
using Random

dir = splitdir(splitdir(pathof(Turing))[1])[1]
include(dir*"/test/test_utils/AllUtils.jl")

@testset "GalacticBridge.jl" begin
    @testset "MLE" begin
        Random.seed!(222)
        true_value = [0.0625, 1.75]
        true_value_named = (s=true_value[1],m=true_value[2])

        f1 = instantiate_galacticoptim_function(gdemo_default, MLE(), unconstrained())
        p1 = GalacticOptim.OptimizationProblem(f1.f, f1.init(true_value), nothing)
        p2 = GalacticOptim.OptimizationProblem(f1.f, f1.init(true_value_named), nothing)

        m1 = solve(p1, NelderMead())
        m2 = solve(p1, LBFGS())
        m3 = solve(p1, BFGS())
        m4 = solve(p2, NelderMead())
        m5 = solve(p2, LBFGS())
        m6 = solve(p2, BFGS())

        @test all(isapprox.(f1.transform(m1.minimizer) - true_value, 0.0, atol=0.01))
        @test all(isapprox.(f1.transform(m2.minimizer) - true_value, 0.0, atol=0.01))
        @test all(isapprox.(f1.transform(m3.minimizer) - true_value, 0.0, atol=0.01))
        @test all(isapprox.(f1.transform(m4.minimizer) - true_value, 0.0, atol=0.01))
        @test all(isapprox.(f1.transform(m5.minimizer) - true_value, 0.0, atol=0.01))
        @test all(isapprox.(f1.transform(m6.minimizer) - true_value, 0.0, atol=0.01))
    end

    @testset "MAP" begin
        Random.seed!(222)
        true_value = [49 / 54, 7 / 6]
        true_value_named = (s=true_value[1],m=true_value[2])

        f1 = instantiate_galacticoptim_function(gdemo_default, MAP(), unconstrained())
        p1 = GalacticOptim.OptimizationProblem(f1.f, f1.init(true_value), nothing)
        p2 = GalacticOptim.OptimizationProblem(f1.f, f1.init(true_value_named), nothing)

        m1 = solve(p1, NelderMead())
        m2 = solve(p1, LBFGS())
        m3 = solve(p1, BFGS())
        m4 = solve(p2, NelderMead())
        m5 = solve(p2, LBFGS())
        m6 = solve(p2, BFGS())

        @test all(isapprox.(f1.transform(m1.minimizer) - true_value, 0.0, atol=0.01))
        @test all(isapprox.(f1.transform(m2.minimizer) - true_value, 0.0, atol=0.01))
        @test all(isapprox.(f1.transform(m3.minimizer) - true_value, 0.0, atol=0.01))
        @test all(isapprox.(f1.transform(m4.minimizer) - true_value, 0.0, atol=0.01))
        @test all(isapprox.(f1.transform(m5.minimizer) - true_value, 0.0, atol=0.01))
        @test all(isapprox.(f1.transform(m6.minimizer) - true_value, 0.0, atol=0.01))
    end

    @testset "MLE constrained" begin
        Random.seed!(222)
        true_value = [0.0625, 1.75]
        true_value_named = (s=true_value[1],m=true_value[2])
        lb = [0.0, 0.0]
        ub = [2.0, 2.0]

        f1 = instantiate_galacticoptim_function(gdemo_default, MLE(), constrained())
        p1 = GalacticOptim.OptimizationProblem(f1.f, f1.init(true_value), nothing; lb=lb, ub=ub)
        p2 = GalacticOptim.OptimizationProblem(f1.f, f1.init(true_value_named), nothing; lb=lb, ub=ub)

        m1 = solve(p1, BBO())
        m2 = solve(p1, Fminbox(LBFGS()))
        m3 = solve(p1, Fminbox(BFGS()))
        m4 = solve(p2, BBO())
        m5 = solve(p2, Fminbox(LBFGS()))
        m6 = solve(p2, Fminbox(BFGS()))

        @test all(isapprox.(f1.transform(m1.minimizer) - true_value, 0.0, atol=0.01))
        @test all(isapprox.(f1.transform(m2.minimizer) - true_value, 0.0, atol=0.01))
        @test all(isapprox.(f1.transform(m3.minimizer) - true_value, 0.0, atol=0.01))
        @test all(isapprox.(f1.transform(m4.minimizer) - true_value, 0.0, atol=0.01))
        @test all(isapprox.(f1.transform(m5.minimizer) - true_value, 0.0, atol=0.01))
        @test all(isapprox.(f1.transform(m6.minimizer) - true_value, 0.0, atol=0.01))
    end

    @testset "MAP constrained" begin
        Random.seed!(222)
        true_value = [49 / 54, 7 / 6]
        true_value_named = (s=true_value[1],m=true_value[2])
        lb = [0.0, 0.0]
        ub = [2.0, 2.0]

        f1 = instantiate_galacticoptim_function(gdemo_default, MAP(), constrained())
        p1 = GalacticOptim.OptimizationProblem(f1.f, f1.init(true_value), nothing; lb=lb, ub=ub)
        p2 = GalacticOptim.OptimizationProblem(f1.f, f1.init(true_value_named), nothing; lb=lb, ub=ub)

        m1 = solve(p1, BBO())
        m2 = solve(p1, Fminbox(LBFGS()))
        m3 = solve(p1, Fminbox(BFGS()))
        m4 = solve(p2, BBO())
        m5 = solve(p2, Fminbox(LBFGS()))
        m6 = solve(p2, Fminbox(BFGS()))

        @test all(isapprox.(f1.transform(m1.minimizer) - true_value, 0.0, atol=0.01))
        @test all(isapprox.(f1.transform(m2.minimizer) - true_value, 0.0, atol=0.01))
        @test all(isapprox.(f1.transform(m3.minimizer) - true_value, 0.0, atol=0.01))
        @test all(isapprox.(f1.transform(m4.minimizer) - true_value, 0.0, atol=0.01))
        @test all(isapprox.(f1.transform(m5.minimizer) - true_value, 0.0, atol=0.01))
        @test all(isapprox.(f1.transform(m6.minimizer) - true_value, 0.0, atol=0.01))
    end
end