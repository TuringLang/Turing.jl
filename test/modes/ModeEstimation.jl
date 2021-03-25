

@testset "ModeEstimation.jl" begin
    @testset "MLE" begin
        Random.seed!(222)
        true_value = [0.0625, 1.75]

        f1 = galacticoptim_function(gdemo_default, MLE())
        p1 = GalacticOptim.OptimizationProblem(f1.f, f1.init(true_value), nothing)

        p2 = optim_problem(gdemo_default, MLE())
        

        m1 = solve(p1, NelderMead())
        m2 = solve(p1, LBFGS())
        m3 = solve(p1, BFGS())
        m4 = optimize(p2.obj, (G,z) -> p2.obj(nothing,G,nothing,z), p2.init(true_value), NelderMead())
        m5 = optimize(p2.obj, (G,z) -> p2.obj(nothing,G,nothing,z), p2.init(true_value), LBFGS())
        m6 = optimize(p2.obj, (G,z) -> p2.obj(nothing,G,nothing,z), p2.init(true_value), BFGS())

        @test all(isapprox.(f1.transform(m1.minimizer) - true_value, 0.0, atol=0.01))
        @test all(isapprox.(f1.transform(m2.minimizer) - true_value, 0.0, atol=0.01))
        @test all(isapprox.(f1.transform(m3.minimizer) - true_value, 0.0, atol=0.01))
        @test all(isapprox.(p2.transform(m4.minimizer) - true_value, 0.0, atol=0.01))
        @test all(isapprox.(p2.transform(m5.minimizer) - true_value, 0.0, atol=0.01))
        @test all(isapprox.(p2.transform(m6.minimizer) - true_value, 0.0, atol=0.01))
    end

    @testset "MAP" begin
        Random.seed!(222)
        true_value = [49 / 54, 7 / 6]

        f1 = galacticoptim_function(gdemo_default, MAP())
        p1 = GalacticOptim.OptimizationProblem(f1.f, f1.init(true_value), nothing)

        p2 = optim_problem(gdemo_default, MAP())
        

        m1 = solve(p1, NelderMead())
        m2 = solve(p1, LBFGS())
        m3 = solve(p1, BFGS())
        m4 = optimize(p2.obj, (G,z) -> p2.obj(nothing,G,nothing,z), p2.init(true_value), NelderMead())
        m5 = optimize(p2.obj, (G,z) -> p2.obj(nothing,G,nothing,z), p2.init(true_value), LBFGS())
        m6 = optimize(p2.obj, (G,z) -> p2.obj(nothing,G,nothing,z), p2.init(true_value), BFGS())

        @test all(isapprox.(f1.transform(m1.minimizer) - true_value, 0.0, atol=0.01))
        @test all(isapprox.(f1.transform(m2.minimizer) - true_value, 0.0, atol=0.01))
        @test all(isapprox.(f1.transform(m3.minimizer) - true_value, 0.0, atol=0.01))
        @test all(isapprox.(p2.transform(m4.minimizer) - true_value, 0.0, atol=0.01))
        @test all(isapprox.(p2.transform(m5.minimizer) - true_value, 0.0, atol=0.01))
        @test all(isapprox.(p2.transform(m6.minimizer) - true_value, 0.0, atol=0.01))
    end

    @testset "MLE constrained" begin
        Random.seed!(222)
        true_value = [0.0625, 1.75]
        lb = [0.0, 0.0]
        ub = [2.0, 2.0]

        f1 = galacticoptim_function(gdemo_default, MLE(true))
        p1 = GalacticOptim.OptimizationProblem(f1.f, f1.init(true_value), nothing; lb=lb, ub=ub)

        p2 = optim_problem(gdemo_default, MLE(true))
        

        m1 = solve(p1, Fminbox(LBFGS()))
        m2 = solve(p1, Fminbox(BFGS()))
        m3 = optimize(p2.obj, (G,z) -> p2.obj(nothing,G,nothing,z),lb, ub, p2.init(true_value), Fminbox(LBFGS()))
        m4 = optimize(p2.obj, (G,z) -> p2.obj(nothing,G,nothing,z), lb, ub, p2.init(true_value), Fminbox(BFGS()))


        @test all(isapprox.(f1.transform(m1.minimizer) - true_value, 0.0, atol=0.01))
        @test all(isapprox.(f1.transform(m2.minimizer) - true_value, 0.0, atol=0.01))
        @test all(isapprox.(f1.transform(m3.minimizer) - true_value, 0.0, atol=0.01))
        @test all(isapprox.(p2.transform(m4.minimizer) - true_value, 0.0, atol=0.01))
    end

    @testset "MAP constrained" begin
        Random.seed!(222)
        true_value = [49 / 54, 7 / 6]
        lb = [0.0, 0.0]
        ub = [2.0, 2.0]

        f1 = galacticoptim_function(gdemo_default, MAP(true))
        p1 = GalacticOptim.OptimizationProblem(f1.f, f1.init(true_value), nothing; lb=lb, ub=ub)

        p2 = optim_problem(gdemo_default, MAP(true))
        

        m1 = solve(p1, Fminbox(LBFGS()))
        m2 = solve(p1, Fminbox(BFGS()))
        m3 = optimize(p2.obj, (G,z) -> p2.obj(nothing,G,nothing,z),lb, ub, p2.init(true_value), Fminbox(LBFGS()))
        m4 = optimize(p2.obj, (G,z) -> p2.obj(nothing,G,nothing,z), lb, ub, p2.init(true_value), Fminbox(BFGS()))


        @test all(isapprox.(f1.transform(m1.minimizer) - true_value, 0.0, atol=0.01))
        @test all(isapprox.(f1.transform(m2.minimizer) - true_value, 0.0, atol=0.01))
        @test all(isapprox.(f1.transform(m3.minimizer) - true_value, 0.0, atol=0.01))
        @test all(isapprox.(p2.transform(m4.minimizer) - true_value, 0.0, atol=0.01))
    end
end