using FileIO
using Images

img_names = readdir("/home/kai/tmp/1")

gen_hist(x::Vector, nbin::Int) = begin
  max_x = 1.25
  min_x = -0.25
  step = (max_x - min_x) / nbin
  bins = zeros(Int, nbin)
  for val in x
    idx = ceil(Int, (val - min_x) / step)
    idx = idx == 0 ? 1 : idx
    bins[idx] += 1
  end
  bins
end

trapz{Tx<:Number, Ty<:Number}(x::Vector{Tx}, y::Vector{Ty}) = begin
    # Trapezoidal integration rule
    local n = length(x)
    if (length(y) != n)
        error("Vectors 'x', 'y' must be of same length")
    end
    r = zero(zero(Tx) + zero(Ty))
    if n == 1; return r; end
    for i in 2:n
        r += (x[i] - x[i-1]) * (y[i] + y[i-1])
    end
    #= correction -h^2/12 * (f'(b) - f'(a))
    ha = x[2] - x[1]
    he = x[end] - x[end-1]
    ra = (y[2] - y[1]) / ha
    re = (y[end] - y[end-1]) / he
    r/2 - ha*he/12 * (re - ra)
    =#
    return r/2
end

make_vec(img::Vector) = begin
  nbin = 192
  x = zeros(Float64, nbin)
  x[end] = 1.0
  max_x = 1
  min_x = 0
  step = (max_x - min_x) / nbin
  for i = 2:191
    x[i] = x[i-1] + step
  end
  y = gen_hist(img, nbin)
  y, y./trapz(x,y)
end

img = load("/home/kai/tmp/1/$(img_names[1])")
ch1 = vec(map(rgb -> getfield(rgb,1), img.data))
h, h_n = make_vec(ch1)

using Gadfly
l1 = layer(x=ch1, Geom.density)
nbin = 192
x = zeros(Float64, nbin)-0.25
x[end] = 1.25
max_x = 1.25
min_x = -0.25
step = (max_x - min_x) / nbin
for i = 2:191
  x[i] = x[i-1] + step
end
l2 = layer(x=x,y=h_n,Geom.line)
plot(l1, l2)

N = 1000
i = 1
j = 1
x = []
while i <= N
  print(i, " ")
  img = load("/home/kai/tmp/1/$(img_names[j])")
  if isa(img.data[1], Images.RGB)
    ch1 = vec(map(rgb -> getfield(rgb,1), img.data))
    ch2 = vec(map(rgb -> getfield(rgb,2), img.data))
    ch3 = vec(map(rgb -> getfield(rgb,3), img.data))
    concat_hist = [normalize(gen_hist(ch1, 192)); normalize(gen_hist(ch2, 192)); normalize(gen_hist(ch3, 192))]
    push!(x, concat_hist)
    i += 1
  end
  j += 1
end

data = Dict()
data["N"] = N
data["x"] = x
