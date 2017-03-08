using Distributions, Turing

N = 10
y = [0, 1, 0, 1, 0, 0, 0, 0, 0, 1]

@model bernoulli(y) begin
  p ~ Beta(1,1)
  for i =1:N
    y[i] ~ Bernoulli(p)
  end
  return(p)
end

c = @sample(bernoulli(y), HMC(1000, 0.2, 5))

t = 0
for _ = 1:10
  t += @elapsed @sample(bernoulli(y), HMC(1000, 0.2, 5))
end
t / 10  # => 8.04s Mon 6 Mar 15:16:40
        # => 7.42s Tue 7 Mar 12:59:55 (a083e820c26f7a02e62d0d24f45890d774940cca)
        # => 6.25s Wed 8 Mar 11:01:55 (3d59ac810d7cca5ff642ff404e80e982c5173f27)
