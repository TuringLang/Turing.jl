using ForwardDiff: Dual

function f(a, b)
  return a * b
end

a = Dual(1, 1, 0)
b = Dual(2, 0, 1)

f(a, b)

d = Dual(1)
d.value
d.partials.values
d = Dual{2, Float64}(d.value)


a = Dual(1)
b = Dual{1, Float64}(2)
a+b
