include("hmc.jl")

##########################################
# Task 0 - implement function f()        #
##########################################
# Definition of f():                     #
# f() is the joint pdf of v and xs       #
# , where v ~ N(0, 3)                    #
#     and x ~ N(0, sqrt(e(v)) (for x in xs) #
##########################################

function f(agrList::Vector)
  v = agrList[1]            # fetch v from the input vector
  xs = agrList[2:end]       # fetch xs from the input vector
  p = 1                     # initialization of pdf
  p *= N(0, 3, v)           # v ~ N(0, 3)
  for x in xs
    p *= N(0, sqrt(exp(v)), x)   # x ~ N(0, sqrt(e(v))
  end
  return p
end

# TODO: solve Problem 1
# Problem 1:
# I can't use pdf(Distribution, x) in f() for some reasone.
# Need to figure out why the following fails
# function f(agrList::Vector)
#   v = agrList[1]            # fetch v from the input vector
#   xs = agrList[2:end]       # fetch xs from the input vector
#   p = 1                               # initialization of pdf
#   p *=  pdf(Normal(0, 3), v)           # v ~ N(0, 3)
#   for x in xs
#     p *= pdf(Normal(0, sqrt(e(v))), x)   # x ~ N(0, sqrt(e(v))
#   end
#   return p
# end

# get the graident ∇f
∇f = ForwardDiff.gradient(f)

# Test f() and its gradient
f(Float64[1, 1, 2])
∇f(Float64[1, 2, 2])

# A simpler f() - multivariate Gaussian
μ = [3.0, 3.0]
Σ = [1.0 0.0;
     0.0 1.0]
Λ = inv(Σ)

function simplef(x::Vector)
  return 1 / sqrt((2pi) ^ 2 * det(Σ)) * exp(-0.5 * ((x - μ)' * Λ * (x - μ))[1])
end

function simplef(x::Vector)
  return hmcpdf(hmcMvNormal(μ, Σ), x)
end

# Test f() and its gradient
simplef([3.0, 3.0])
∇simplef = ForwardDiff.gradient(simplef)

function true∇simplef(x::Vector)
  return 1 / sqrt((2pi) ^ 2 * det(Σ)) * -exp(-0.5 * ((x - μ)' * Λ * (x - μ))[1]) * Λ * (x - μ)
end

∇simplef([2.0, 2.0])
true∇simplef([2.0, 2.0])

#############################################
# Task 1 - implement HMC to sample from f() #
#############################################

# sample from simplef()
samples = HMCSampler(simplef, 500, 0.05, 20, 2)
eval2DSamples(samples)

# sample from f()
samples = HMCSampler(f, 500, 0.01, 20, 2)

##############################################
# Task 2 - compute the effective sample size #
##############################################

sampleNum = 1000
HMCsamples = HMCSampler(simplef, sampleNum, 0.05, 20, 2)
MHSamples = MHSampler(simplef, sampleNum, 0.5, 2)
ess(HMCsamples)
ess(MHSamples)

###############################################
# Task 3 - use meta-programming to facilitate #
#  the HMC sampler and @model to do sampling  #
###############################################

####################
# Demo for meeting #
#  on 16/05/2016   #
####################

# A simple multivariate Gaussian
μ = [3.0, 3.0]
Σ = [1.0 0.5;
     0.5 1.0]
Λ = inv(Σ)

function simplef(x::Vector)
  return 1 / sqrt((2pi) ^ 2 * det(Σ)) * exp(-0.5 * ((x - μ)' * Λ * (x - μ))[1])
end

# Demo 1
function demo1(sampleNum::Int64)
  # MH
  MHSamples = MHSampler(simplef, sampleNum, 0.5, 2)
  MHSampleLayer = layer(x=Float64[x[1] for x in MHSamples], y=Float64[x[2] for x in MHSamples], Geom.point, Theme(default_color=colorant"green"))

  # HMC
  HMCSamples = HMCSampler(simplef, sampleNum, 0.05, 20, 2)
  HMCSampleLayer = layer(x=Float64[x[1] for x in HMCSamples], y=Float64[x[2] for x in HMCSamples], Geom.point, Theme(default_color=colorant"red"))

  # Exact
  exactSampleLayer = layer(z=(x,y) -> simplef([x, y]), x=linspace(-2,8,100), y=linspace(-2,8,100), Geom.contour(levels=5))

  # plot together with real dneisty
  plot(MHSampleLayer, HMCSampleLayer, exactSampleLayer, Guide.xlabel("dim 1"), Guide.ylabel("dim 2"), Guide.title("Samples using MH and HMC"), Coord.cartesian(xmin=-2, xmax=8, ymin=-2, ymax=8), Guide.manual_color_key("Legend", ["MH", "HMC"], ["green", "red"]))
end

demo1(25)

# Demo 2

# Exact
exactSampleLayer = layer(z=(x,y) -> simplef([x, y]), x=linspace(-2,8,100), y=linspace(-2,8,100), Geom.contour(levels=3))

# plot MH with order
function demo21(sampleNum::Int64)
  MHSamples2 = MHSampler(simplef, sampleNum, 0.5, 2)
  MHSampleLayer2 = layer(x=Float64[x[1] for x in MHSamples2], y=Float64[x[2] for x in MHSamples2], Geom.point, color=1:sampleNum+1)
  plot(MHSampleLayer2, exactSampleLayer, Guide.xlabel("dim 1"), Guide.ylabel("dim 2"), Guide.title("Samples using MH with order shown"), Coord.cartesian(xmin=-2, xmax=8, ymin=-2, ymax=8))
end

demo21(55)

# plot HMC with order
function demo22(sampleNum::Int64)
  HMCSamples2 = HMCSampler(simplef, sampleNum, 0.05, 20, 2)
  HMCSampleLayer2 = layer(x=Float64[x[1] for x in HMCSamples2], y=Float64[x[2] for x in HMCSamples2], color=1:sampleNum+1, Geom.point)
  plot(HMCSampleLayer2, exactSampleLayer, Guide.xlabel("dim 1"), Guide.ylabel("dim 2"), Guide.title("Samples using HMC with order shown"), Coord.cartesian(xmin=-2, xmax=8, ymin=-2, ymax=8))
end

demo22(25)
