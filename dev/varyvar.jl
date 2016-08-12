using Turing, Distributions, DualNumbers

M = 10           # number of means
xs = [1.5, 2.0] # the observations
@model gauss_var begin
  ms = Vector{Dual}(M) # initialise an array to store means
  for i = 1:M
    @assume ms[i] ~ Normal(0, sqrt(2))  # define the mean
  end
  for i = 1:length(xs)
  	@observe xs[i] ~ Normal(mean(ms), sqrt(2)) # observe data points
  end
  @predict ms                  # ask predictions of s and m
end

# @time chain = sample(gauss_var, HMC(1000, 0.55, 5))



t1 = time()
for _ = 1:10
  sample(gauss_var, HMC(250, 0.45, 5))
end
t2 = time()
t = (t2 - t1) / 10


# times = []

# push!(times, t)

ts = [0.59798, 1.09218, 2.03847, 3.25326, 4.93184, 6.75097, 8.27073, 11.0466, 13.656764888763428]
ms = 1:9

using Gadfly
hmc_layer = layer(x=ms, y=ts, Geom.line)

p = plot(hmc_layer, Guide.ylabel("Time used (s)"), Guide.xlabel("#variables (n)"))

draw(PDF("/Users/kai/Turing/docs/report/varyvar.pdf", 4inch, 4inch), p)

# varying number of  samples

sample_nums = [10, 100, 500, 1000, 2000, 5000, 10000]
hmc_time_1 = [0.004650497436523437,0.04269888401031494,0.20954039096832275,0.4710076093673706,0.938780689239502,2.242368292808533,3.5772396087646485]
hmc_time_2 = [0.003940796852111817,0.05217530727386475,0.24969069957733153,0.5556180000305175,1.0059500932693481,2.4296218156814575,4.835002803802491]
hmc_time_3 = [0.011547994613647462,0.13400509357452392,0.6510027170181274,1.0813661098480225,2.6234392881393434, 5.46103, 10.5817 ]

using Gadfly
hmc_layer_1 = layer(x=sample_nums, y=hmc_time_1, Geom.line, Theme(default_color=colorant"seagreen"))
hmc_layer_2 = layer(x=sample_nums, y=hmc_time_2, Geom.line, Theme(default_color=colorant"springgreen"))
hmc_layer_3 = layer(x=sample_nums, y=hmc_time_3, Geom.line, Theme(default_color=colorant"violet"))

p = plot(hmc_layer_1, hmc_layer_2, hmc_layer_3, Guide.ylabel("Time used (s)"), Guide.xlabel("#samples (n)"), Guide.manual_color_key("Legend", ["gauss", "beta", "lr"], ["seagreen", "springgreen", "violet"]))

draw(PDF("/Users/kai/Turing/docs/report/varysample.pdf", 4inch, 4inch), p)


eps = [0.1, 0.3, 0.5]
ess_s_n1 = [11.9, 25.8, 47.1, 110.7]
ess_m_n1 = [10.8, 49.3, 102.6, 326]

ess_s_n2 = [15.9, 51.3, 97.1, 129.4]
ess_m_n2 = [26.3, 178.4, 454.26, 502.2]

ess_s_n5 = [49.6, 172.6, 189.1]
ess_m_n5 = [135.1, 773.4, 876.6]


layer_1 = layer(x=eps, y=ess_s_n5, Geom.line, Theme(default_color=colorant"seagreen"))
layer_2 = layer(x=eps, y=ess_m_n5, Geom.line, Theme(default_color=colorant"royalblue"))

p = plot(layer_1, layer_2, Guide.ylabel("ESS"), Guide.xlabel("'leapfrog' step size"), Guide.manual_color_key("Legend", ["s", "m"], ["seagreen", "royalblue"]))

draw(PDF("/Users/kai/Turing/docs/report/diffn5.pdf", 4inch, 4inch), p)



macro change_operation(ex)
  # Change the operation to multiplication
  ex.args[1] = :*
  # Return an expression to print the result
  return :(println($(ex)))
end

ex = @change_operation 1 + 2

macroexpand(:(ex))
