using Gadfly, Cairo             # for plotting
import ForwardDiff              # for graident
include("hmcDistributions.jl")
include("hmc.jl")

function f2(args::Vector)
  μ = args[1]
  σ = args[2]
  pdf = hmcNormal(μ, σ)
  xs = [1, 1, 1, 1.25, 0.75]
  ll = 0
  for x in xs
    ll += log(hmcpdf(pdf, x))
  end
  return ll
end

f2(Float64[0, 1])

∇f2 = ForwardDiff.gradient(f2)


∇f2(Float64[0, 1])

HMCSamples = HMCSampler(f2, 500, 0.04, 10, 2)
HMCSamples = sampleTransform(HMCSamples)

function fun3(args::Vector)
  μ = args[1]
  σ = args[2]
  pdf = hmcNormal(μ, σ)
  xs = randn(100)
  l = map(pdf, xs)
  return prod(l)
end

HMCSamples = HMCSampler(fun3, 1000, 0.01, 10, 2)
eval2DSamples(HMCSamples)

plot(z=(x,y) -> fun3([x,y]),
     x=linspace(-0.5,0.5,50), y=linspace(-1,1,50), Geom.contour)


# TODO: MoN
