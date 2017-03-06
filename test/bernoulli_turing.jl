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
