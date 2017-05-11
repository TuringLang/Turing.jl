@model gaussmodel(x) = begin
    N = length(x)
    lam = tzeros(Real,N)
    mu ~ Normal(0, sqrt(1000));
    for i = 1:N
      lam[i] ~ Gamma(1, 1)
      x[i] ~ Normal(mu, sqrt(1 / (sqrt(lam[i]))))
    end
    return(mu, lam)
end
