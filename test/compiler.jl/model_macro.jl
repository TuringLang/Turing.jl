using Turing, Distributions, Test

@model testmodel(x1, x2) = begin
    s ~ InverseGamma(2,3)
    m ~ Normal(0,sqrt(s))
    x1 ~ Normal(m, sqrt(s))
    x2 ~ Normal(m, sqrt(s))
	return x1, x2
end



f = testmodel(1., 10.)
