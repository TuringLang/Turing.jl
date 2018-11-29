using Turing: WelfordCovar, NaiveCovar, add_sample!, get_covar, reset!
using Test, LinearAlgebra

let
    D = 1000
    wc = WelfordCovar(0, zeros(D), zeros(D,D))
    add_sample!(wc, randn(D))
    reset!(wc)

    # Check that reseting zeros everything.
    @test wc.n === 0
    @test wc.μ == zeros(D)
    @test wc.M == zeros(D,D)

    # Ensure that asking for the variance doesn't mutate the WelfordVar.
    add_sample!(wc, randn(D))
    add_sample!(wc, randn(D))
    μ, M = deepcopy(wc.μ), deepcopy(wc.M)

    @test wc.μ == μ
    @test wc.M == M
end

# Check that the estimated variance is approximately correct.
let
    D = 10
    wc = WelfordCovar(0, zeros(D), zeros(D,D))

    for _ = 1:10000
        s = randn(D)
        add_sample!(wc, s)
    end

    covar = get_covar(wc)

    @test covar ≈ LinearAlgebra.diagm(0 => ones(D)) atol=0.2
end
