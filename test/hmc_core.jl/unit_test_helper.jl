using Test

function test_grad(turing_model, grad_f; trans=Dict())
    model_f = turing_model()
    vi = model_f()
    for i in trans
        vi.flags["trans"][i] = true
    end
    d = length(vi.vals)
    @testset "Gradient using random inputs" begin
        for _ = 1:10000
            theta = rand(d)
            @test Turing.gradient_reverse(theta, vi, model_f) == grad_f(theta)[2]
        end
    end
end
