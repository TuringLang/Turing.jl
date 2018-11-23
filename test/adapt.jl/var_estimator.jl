using Turing: WelfordVar, add_sample!, get_var, reset!
using Test

let
    D = 1000
    wv = WelfordVar(0, zeros(D), zeros(D))
    add_sample!(wv, randn(D))
    reset!(wv)

    # Check that reseting zeros everything.
    @test wv.n === 0
    @test wv.μ == zeros(D)
    @test wv.M == zeros(D)

    # Ensure that asking for the variance doesn't mutate the WelfordVar.
    add_sample!(wv, randn(D))
    add_sample!(wv, randn(D))
    μ, M = deepcopy(wv.μ), deepcopy(wv.M)
    get_var(wv)
    @test wv.μ == μ
    @test wv.M == M
end

# Check that the estimated variance is approximately correct.
let
    D = 10
    wv = WelfordVar(0, zeros(D), zeros(D))

    for _ = 1:10000
        s = randn(D)
        add_sample!(wv, s)
    end

    var = get_var(wv)

    @test var ≈ ones(D) atol=0.5
end
