Pkg.add("ForwardDiff")
import ForwardDiff

f(x::Vector) = sum(sin, x) + prod(tan, x) * sum(sqrt, x);
g = ForwardDiff.gradient(f);
# x = rand(5)
x = [0.986403, 0.140913, 0.294963, 0.837125, 0.650451]
g(x)

h = ForwardDiff.hessian(f)
h(x)
