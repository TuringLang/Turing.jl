@model gaussmodel(x) = begin
    N = length(x)
    lam = ones(Float64,N)
    mu ~ Normal(0, sqrt(1000));
    for i = 1:N
      lam[i] ~ Gamma(.001, .001)
      x[i] ~ Normal(mu, sqrt(1 / (sqrt(lam[i]))))
    end
    return(mu, lam)
end
