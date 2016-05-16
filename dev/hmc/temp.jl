import ForwardDiff          # for graident

μ = [3.0, 3.0]
Σ = [1.0 0.0;
     0.0 1.0]
Λ = inv(Σ)

function f(x::Vector)
  return 1 / sqrt((2pi) ^ 2 * det(Σ)) * exp(-0.5 * ((x - μ)' * Λ * (x - μ))[1])
end

x = [3.0, 3.0]
f(x)

function g(x::Vector)
  return 1 / sqrt((2pi) ^ 2 * det(Σ)) * -exp(-0.5 * ((x - μ)' * Λ * (x - μ))[1]) * Λ * (x - μ)
end




function f3(x::Vector)
  return 1 / sqrt(2pi) * exp(-0.5 * x[1]^2)
end

g3 = ForwardDiff.gradient(f3)
x = [1.0]

g3(x)

M = 200
d = 1
L = 10
ss = 0.01
samples = Array(typeof(x), M)



th = [2.0]
accCount = 0

m = 1

p = randn(d)[1]
H = f3(th) + p * p / 2
th2 = th

# println(g(th2))
p -= ss / 2 * g3(th2)
th2 += ss * p
p -= ss / 2 * g3(th2)


th2
p

H2 = f3(th2) + p' * p / 2
println(H - H2)
if rand() < min(1.0, exp(H - H2)[1])
  th = th2
  accCount += 1
end
accCount
th
samples[m] = th

m += 1

samples
println(accCount)
eval2DSamples(samples)
