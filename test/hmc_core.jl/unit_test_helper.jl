using Base.Test

function test_grad(turing_model, grad_f)
    model_f = turing_model()
    vi = Base.invokelatest(model_f)
    d = length(vi.vals)
    @testset "Gradient using random inputs" begin
        for _ = 1:10000
            theta = rand(d)
            @test Turing.gradient_r(theta, vi, model_f) == grad_f(theta)[2]
        end
    end
end