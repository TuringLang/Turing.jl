using Turing
using Mamba: describe

include(Pkg.dir("Turing")*"/example-models/stan-models/school8-stan.data.jl")

@model school8(J, y, sigma) = begin
  mu ~ NoInfo()
  tau ~ NoInfoPos(0)
  eta ~ MvNormal(zeros(J), ones(J))
  y ~ MvNormal(mu .+ tau .* eta, sigma)
end

data = deepcopy(schools8data[1])
delete!(data, "tau")

chn = sample(school8(data=data), HMC(2000, 0.75, 5))

describe(chn)
