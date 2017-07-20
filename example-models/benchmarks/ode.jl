odeDataRaw = readcsv(Pkg.dir("Turing")*"/example-models/benchmarks/ode.csv")
t = Vector{Float64}(odeDataRaw[2:end,1])
x0 = Vector{Float64}(odeDataRaw[2:end,2])
x1 = Vector{Float64}(odeDataRaw[2:end,3])

sho(t, y, theta, x_r, x_i) = begin
  dydt = Vector{Real}(2)
  dydt[1] =
end
{
  real dydt[2];
  dydt[1] <- y[2];
  dydt[2] <- -y[1] - theta[1] * y[2];
  return dydt;
 }
