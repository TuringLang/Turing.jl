# To run the Bernoulli example, start by concatenating the home directory and project directory:
using Mamba, Stan

old = pwd()
ProjDir = Pkg.dir("Stan", "Examples", "Bernoulli")
cd(ProjDir)

# Next define the variable 'bernoullistanmodel' to hold the Stan model definition:
const bernoullistanmodel = "
data {
  int<lower=0> N;
  int<lower=0,upper=1> y[N];
}
parameters {
  real<lower=0,upper=1> theta;
}
model {
  theta ~ beta(1,1);
    y ~ bernoulli(theta);
}
"

# The next step is to create a Stanmodel object. The most common way to create such an object is by giving the model a name while the Stan model is passed in, both through keyword (hence optional) arguments:
stanmodel = Stanmodel(name="bernoulli", model=bernoullistanmodel);

# The input data is defined below. By default 4 chains will be simulated. Below initialization of 'bernoullidata' creates an array of 4 dictionaries, a dictionary for each chain. If the array length is not equal to the number of chains, only the first element of the array will be used as initialization for all chains.
const bernoullidata = [
  Dict("N" => 10, "y" => [0, 1, 0, 1, 0, 0, 0, 0, 0, 1]),
  Dict("N" => 10, "y" => [0, 1, 0, 0, 0, 0, 1, 0, 0, 1]),
  Dict("N" => 10, "y" => [0, 1, 0, 0, 0, 0, 0, 0, 1, 1]),
  Dict("N" => 10, "y" => [0, 0, 0, 1, 0, 0, 0, 1, 0, 1])
]
println("Input observed data, an array of dictionaries:")
bernoullidata |> display
println()

# Run the simulation by calling stan(), passing in the data and the intended working directory. To get a summary description of the results, describe() is called (describe() is a Mamba.jl function):
sim1 = stan(stanmodel, bernoullidata, ProjDir, CmdStanDir=CMDSTAN_HOME)
describe(sim1)

# In this case 'sim1' is a Mamba Chains object. We can inspect sim1 as follows:
typeof(sim1) |> display
fieldnames(sim1) |> display
sim1.names |> display

# To inspect the simulation results by Mamba's describe() we can't use all monitored variables by Stan. In this example a good subset is selected as follows and stored in 'sim':
println("Subset Sampler Output")
sim = sim1[1:1000, ["lp__", "theta", "accept_stat__"], :]
describe(sim)
println()

# The following diagnostics and Gadfly based plot functions (all from Mamba.jl) are available:
println("Brooks, Gelman and Rubin Convergence Diagnostic")
try
  gelmandiag(sim1, mpsrf=true, transform=true) |> display
catch e
  #println(e)
  gelmandiag(sim, mpsrf=false, transform=true) |> display
end
println()

println("Geweke Convergence Diagnostic")
gewekediag(sim) |> display
println()

println("Highest Posterior Density Intervals")
hpd(sim) |> display
println()

println("Cross-Correlations")
cor(sim) |> display
println()

println("Lag-Autocorrelations")
autocor(sim) |> display
println()

# To plot the simulation results:
p = plot(sim, [:trace, :mean, :density, :autocor], legend=true);
draw(p, ncol=4, filename="summaryplot", fmt=:svg)
draw(p, ncol=4, filename="summaryplot", fmt=:pdf)

# On OSX, if e.g. JULIA_SVG_BROWSER="Google's Chrome.app" is exported as an environment variable, the .svg files can be displayed as follows:
if length(JULIA_SVG_BROWSER) > 0
  @osx ? for i in 1:3
    isfile("summaryplot-$(i).svg") &&
      run(`open -a $(JULIA_SVG_BROWSER) "summaryplot-$(i).svg"`)
  end : println()
end

cd(old)
