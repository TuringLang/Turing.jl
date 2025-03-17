using Test

function test_grad(turing_model, grad_f; trans=Dict())
    model_f = turing_model()
    vi = model_f()
    for i in trans
        vi.flags["trans"][i] = true
    end
    d = length(vi.vals)
    @testset "Gradient using random inputs" begin
        ℓ = LogDensityProblemsAD.ADgradient(
            Turing.AutoTracker(),
            DynamicPPL.LogDensityFunction(
                model_f,
                vi,
                DynamicPPL.SamplingContext(SampleFromPrior(), DynamicPPL.DefaultContext())
            ),
        )
        for _ in 1:10000
            theta = rand(d)
            @test LogDensityProblems.logdensity_and_gradient(ℓ, theta) == grad_f(theta)[2]
        end
    end
end
