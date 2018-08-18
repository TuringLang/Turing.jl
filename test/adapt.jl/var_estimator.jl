using Turing: VarEstimator, add_sample!, get_var
using Test

D = 10
ve = VarEstimator{Float64}(0, zeros(D), zeros(D))

for _ = 1:10000
    s = randn(D)
    add_sample!(ve, s)
end

var = get_var(ve)

@test var ≈ ones(D) atol=0.5
