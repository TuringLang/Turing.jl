#############################################
## Replicate the Mixtue of Gassusian model ##
##  from the paper by Rich and Yingzheng   ##
#############################################
##       coded by Kai Xu 04/05/2016        ##
#############################################


using Turing, Distributions # for inference & distribution
using Gadfly, Cairo         # for plotting


######################
## Helper functions ##
######################
function synData(N, J, D)
  """
  Generate a synthetic Mixture of Gaussians (MoG) dataset with N datapoints and J Gaussians in D dimension
  N: number of data points to be generated
  J: number of mixtures
  D: dimensionality
  """
  # The means were sampled from a Gaussian distribution:
  #   p(μ_j) = N(μ; m, I) (m = 0 for convenience)
  dμ = MvNormal(Float64[0 for i = 1:D], Float64[1 for i = 1:D]) # distribution of mixture means (μ)
  μs = [Float64[i for i in rand(dμ)] for j = 1:J]               # sampled mixture means

  # The cluster identity varaibles were sampled from a uniform categroical distribution:
  #   p(h_n = j) = 1 / 4
  dh = Categorical(Float64[1/J for i = 1:J])  # distribution of identity variables (h)
  hs = rand(dh, N)                            # sampled cluster identities

  # The mixture components were isotropic Gaussians:
  #   p(x_n| h_n) = N(x_n; μ_h_n, 0.5^2*I)
  xs = [Float64[j for j in rand(MvNormal(μs[hs[i]], Float64[0.25 for k = 1:D]))] for i = 1:N]

  # Pack data
  data = Dict("xs" => xs, "μs" => μs, "hs" => hs)
  return data
end


###########################
## Synthesis data points ##
###########################
# number of data points and mixtures
N, J, D = 20, 3, 2
data = synData(N, J, D) # synthesis data
xs = data["xs"]         # unpack data
exactμs = data["μs"]
exacths = data["hs"]

# plot 2 dimensions of data for visuliazation
l1 = layer(x=[xs[i][1] for i = 1:N], y=[xs[i][2] for i = 1:N], Geom.point)
plot(l1, Guide.XLabel("dim 1"), Guide.YLabel("dim 2"), Guide.Title("Synthesised 2-dimension Gaussian data points from four mixtures"))


####################################
## Build a MoG model using Turing ##
####################################
@printf "mixture number J : %d\n     dimension D : %d" J D

@model MoG begin
  μs = TArray[zeros(Float64, D) for i = 1:J]
  for i = 1:J
    @assume μs[i] ~ MvNormal(Float64[0 for j = 1:D], Float64[1 for j = 1:D])
  end
  hs = tzeros(Float64, length(xs))
  for i = 1:length(xs)
    @assume hs[i] ~ Categorical(Float64[1/J for j = 1:J])
    # need to force converting TArray type to Vector (the mean Vector)
    @observe xs[i] ~ MvNormal(Float64[μs[hs[i]][j] for j = 1:D], Float64[0.25 for j = 1:D])
  end
  @predict μs hs
end

#############################
## Run SMC to do inference ##
#############################

particalNum = 20
@time chain  = sample(MoG, SMC(particalNum))

# fetch chain
μsChain = chain[:μs]
hsChain = chain[:hs]
estimateμs = mean(μsChain)
estimatehs = Int64[Int64(i) for i in mean(hsChain)]

# print estimate value and exact value
println("Estimate μ:")
println(estimateμs)
println("Exact μ:")
println(exactμs)

# plot estiamted means within orignral data
l2 = layer(x=[estimateμs[i][1] for i = 1:J], y=[estimateμs[i][2] for i = 1:J], Geom.point, Theme(default_color=color("orange")), order=1)
plot(l1, l2, Guide.XLabel("dim 1"), Guide.YLabel("dim 2"), Guide.Title("Synthesised 2-dimension Gaussian data points from four mixtures"))


#############################
## Calculate KL divergence ##
#############################
# Definition of KL:
#   D_KL(P||Q) = ∑ P(i) log(P(i)/Q(i))      (discrete)
#   D_KL(P||Q) = ∫ p(x) log(p(x)/q(x))      (continuous)
# , where P - true, and Q - approx.


function mixtureP(μs, x, h, D=2)
  """
  Compute the probablity of a given data in a mixture model.
    μs  -   the mixture means
    x   -   the data point
    h   -   the identity variable
  and returns the pdf of that data point
  """
  d = MvNormal(μs[h], Float64[1 for i = 1:D])
  return pdf(d, x)
end


function kl(xs, exactμs, exacths, estimateμs, estimatehs)
  """
  Compute the KL divergence
    xs          -   data points
    exactμs     -   exact means
    exacths     -   exact mixture identites
    estimateμs  -   estimated means
    estimatehs  -   estimated mixture identites
  """
  D = 0
  for i = 1:length(xs)
    P = mixtureP(exactμs, xs[i], exacths[i])
    Q = mixtureP(estimateμs, xs[i], estimatehs[i])

    D += P * log2(P / Q)
  end
  return D
end


# call the function to conpute KL
kl(xs, exactμs, exacths, estimateμs, estimatehs)


# plot
plot(x=1:particalNum, y=Ds, Geom.line, Guide.xlabel("Number of iteration"), Guide.ylabel("KL"), Guide.title("KL Divergence along SMC iterations"))




# test

d = MvNormal(Float64[0 for i = 1:D], Float64[1 for i = 1:D])
x = rand(d)
x = Float64[0, 0]
pdf(d, x)


s  = 0.25
obs = [0.25, 0, 0, 0, -0.25]

prior = Normal(0, 1)

@model anglican_gaussian begin
  @assume mean ~ prior
  for i = 1:length(obs)
    @observe obs[i] ~ Normal(mean, s)
  end
  @predict mean
end
