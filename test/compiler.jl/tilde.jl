using Turing
import Turing.translate!

ex = quote
  x = 1
  y = rand()
  y ~ Normal(0,1)
end

res = translate!(:(y~Normal(1,1)))

Base.@assert res.head == :macrocall
Base.@assert res.args == [Symbol("@~"), :y, :(Normal(1, 1))]

res2 = translate!(ex)

Base.@assert res2.args[end].head == :macrocall
Base.@assert res2.args[end].args == [Symbol("@~"), :y, :(Normal(1, 1))]
