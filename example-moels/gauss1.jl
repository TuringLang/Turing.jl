using Turing
using Distributions
using Base.Test

@model gauss1 begin
    N = length(x)
    lam = ones(Float64,N)
    mu ~ Normal(0, sqrt(1000));
    for i = 1:N
        lam[i] ~ Gamma(.001, .001)
        x[i] ~ Normal(mu, sqrt(1 / (sqrt(lam[i]))))
    end
    mu
end

x = [-27.020, 3.570, 8.191, 9.898, 9.603, 9.945, 10.056]
chain = sample(gauss1,  SMC(300))
