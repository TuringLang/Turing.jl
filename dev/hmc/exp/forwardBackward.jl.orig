versioninfo()
# Pkg.update()
# Pkg.add("Gadfly")
# Pkg.add("Cairo")
# Pkg.build("Cairo")

# An implementation of forward-backward algorithm for HMM
println("An implementation of forward-backward algorithm for HMM")

using Gadfly
using Cairo

# example of output plot in PNG format:
pl = plot(x -> cos(x)/x, 5, 25)
# draw(PNG("test.png", 300, 200), pl)

# setting of the HMM example
# initial state
π = [1; 0; 0]
# transition probability
A = [0.1 0 0; 0.8 0.15 0; 0.1 0.85 1]
# emission probabiltiy
B = [0.5 0.1 0; 0.4 0.6 0.2; 0.1 0.3 0.8]

# observatins
o = [1 1 2 3]
# generate observation matrix
O = Any[]
state_num = size(A)[1]
obs_num = size(B)[1]
for i in 1:obs_num
  temp = zeros(obs_num, state_num)
  for j in 1:state_num
    temp[j, j] = B[i, j]
  end
  push!(O, temp)
end

# normalize function
function normalize(p)
  Z = sum(p)
  p / Z
end

# forward pass
function α(t, o, π, A, B)
  f = normalize(O[o[1]] * π)
  for i in 2:t
    f = normalize(O[o[i]] * (A * f))
  end
  return f
end

# backward pass
b = ones(state_num, 1)

o_num = length(o)
function β(t, o, π, A, B)
  b = ones(state_num, 1)
  for i in o_num:-1:t
    b = (O[o[i]] * A) * b
  end
  normalize(b)
end

p = Any[]
for i in 1:o_num
  f = α(i, o, π, A, B)
  b = β(i, o, π, A, B)
  push!(p, normalize(f .* b))
end

println(p)
