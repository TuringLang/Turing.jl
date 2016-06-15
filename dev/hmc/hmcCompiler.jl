include("hmc.jl")

### model start

xs = [1, 1.6, 1, 1.1, 0.9, 1.3, 0.9]

function likelihood(argList::Vector)
  μ = argList[1]
  σ = argList[2]

  pdf = hmcNormal(μ, σ)

  xs = [1, 1, 1, 1.25, 0.75]
  lik = 1

  for x in xs
    lik *= hmcpdf(pdf, x)
  end

  return lik
end

function prior(argList::Vector)
  μ = argList[1]
  σ = argList[2]
  pri = 1 * hmcpdf(hmcInverseGamma(2, 3), σ) * hmcpdf(hmcNormal(0, sqrt(σ)), μ)
  return pri
end

function posterior(argList::Vector)
  post = likelihood(argList) * prior(argList)
  return post
end

### model ends

HMCSamples = HMCSampler(posterior, 500, 0.04, 10, 2)
# eval2DSamples(HMCSamples)
s = sampleTransform(HMCSamples)
l1 = layer(x=1:501, y=s[:,1], Geom.line, Theme(default_color=color("#000099")))
l2 = layer(x=1:501, y=s[:,1], Geom.smooth(method=:loess,smoothing=0.9), Theme(default_color=color("orange")))

l3 = layer(x=1:501, y=s[:,2], Geom.line, Theme(default_color=color("green")))
l4 = layer(x=1:501, y=s[:,2], Geom.smooth(method=:loess,smoothing=0.9), Theme(default_color=color("red")))

p = plot(l4,l3,l2,l1, Coord.cartesian(xmin=0, xmax=500), Guide.xlabel("sample number"), Guide.ylabel("parameter value"),Guide.manual_color_key("Legend", ["avergae σ","σ", "average μ", "μ"], ["red", "green", "orange","blue"]))

draw(PNG("/Users/kai/Turing/docs/poster/figures/gauss2.png", 5inch, 4inch), p)


# @model hmmdemo begin
#   states = TArray(Int, length(data))
#   @assume states[1] ~ initial
#   for i = 2:length(data)
#     @assume states[i] ~ trans[states[i-1]]
#     @observe data[i]  ~ Normal(statesmean[states[i]], 0.4)
#   end
#   @predict states data
# end


D = [1,2,3,4]
y = 0
for x in D
  y += x
end
y
