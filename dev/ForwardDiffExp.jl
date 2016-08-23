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






function test_f(a, b, c, d, e)
  return a * b + c * d - e
end

# method 1

t1 = time()
for _ = 1:2500

g = []
r = test_f(Dual(1, 1), Dual(2), Dual(1), Dual(2), Dual(3))
push!(g, dualpart(r)[1])
r = test_f(Dual(1), Dual(2, 1), Dual(1), Dual(2), Dual(3))
push!(g, dualpart(r)[1])
r = test_f(Dual(1), Dual(2), Dual(1, 1), Dual(2), Dual(3))
push!(g, dualpart(r)[1])
r = test_f(Dual(1), Dual(2), Dual(1), Dual(2, 1), Dual(3))
push!(g, dualpart(r)[1])
r = test_f(Dual(1), Dual(2), Dual(1), Dual(2), Dual(3, 1))
push!(g, dualpart(r)[1])

end

t = time() - t1 # 0.167

# method 2

t1 = time()
for _ = 1:2500

g = []
r = test_f(
  Dual(1, 1, 0, 0, 0, 0),
  Dual(2, 0, 1, 0, 0, 0),
  Dual(1, 0, 0, 1, 0, 0),
  Dual(2, 0, 0, 0, 1, 0),
  Dual(3, 0, 0, 0, 0, 1))
dp = dualpart(r)
for i = 1:5
  push!(g, dp[i])
end

end

t = time() - t1 # 0.086


t1 = time()
a = []
for _ = 1:10000
  push!(a, Dual(1, 1, 0, 0, 0, 0))
end
t = time() - t1


function make_dual2(dim, real, idx)
  z = zeros(dim)
  z[idx] = 1
  return Dual(real, tuple(collect(z)...))
end

t1 = time()
a = []
for _ = 1:10000
  push!(a, make_dual2(5, 1, 1))
end
t = time() - t1





test = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
total_l = length(test)
prior_key_chunks = []
for i = 1:10:total_l
  end_idx = min(i + 9, total_l)
  push!(prior_key_chunks,test[i:min(i + 9, total_l)] )
  println("$i, $end_idx")
end


prior_key_chunks
for key_chunk in prior_key_chunks
  println(key_chunk)
end


all_keys = Any["1"]
total_prior_dim = 1
print(all_keys)
prior_key_chunks = []
for i = 1:10:total_prior_dim
  push!(prior_key_chunks, all_keys[i:min(i + 9, total_prior_dim)])
end
