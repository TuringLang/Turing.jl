using Turing



function sigmoid(t)
  return 1 / (1 + e^(-t))
end

function nn(x, b1, w11, w12, w13, bo, wo)
  # wi = weight_in, wh = weight_hidden, wo = weight_out
  h = tanh([w11' * x + b1[1]; w12' * x + b1[2]; w13' * x + b1[3]])
  return sigmoid((wo' * h)[1] + bo)
end

function predict(x, chain)
  n = length(chain[:b1])
  b1 = chain[:b1]
  w11 = chain[:w11]
  w12 = chain[:w12]
  w13 = chain[:w13]
  bo = chain[:bo]
  wo = chain[:wo]
  return mean([nn(x, b1[i], w11[i], w12[i], w13[i], bo[i], wo[i]) for i in 1:n])
end



xs = Array[[1; 1], [0; 0], [1; 0], [0; 1]]
ts = [1; 1; 0; 0]

alpha = 0.16            # regularizatin term
var = sqrt(1.0 / alpha) # variance of the Gaussian prior
@model bnn begin
  @assume b1 ~ MvNormal([0 ;0; 0], [var 0 0; 0 var 0; 0 0 var])
  @assume w11 ~ MvNormal([0; 0], [var 0; 0 var])
  @assume w12 ~ MvNormal([0; 0], [var 0; 0 var])
  @assume w13 ~ MvNormal([0; 0], [var 0; 0 var])
  @assume bo ~ Normal(0, var)
  @assume wo ~ MvNormal([0; 0; 0], [var 0 0; 0 var 0; 0 0 var])
  for i = rand(1:4, 1)
    y = nn(xs[i], b1, w11, w12, w13, bo, wo)
    @observe ts[i] ~ Bernoulli(y)
  end
  @predict b1 w11 w12 w13 bo wo
end

@time chain = sample(bnn, HMC(3200, 0.4, 8))

[predict(xs[i], chain) for i = 1:4]



using Mamba: Chains, summarystats
s = map(x -> x[1], chain[:wo])
println(summarystats(Chains(s)))



using Gadfly
p_layer = layer(z=(x,y) -> predict([x, y], chain), x=linspace(-1,2,15), y=linspace(-1,2,15), Geom.contour)
plot(p_layer)
