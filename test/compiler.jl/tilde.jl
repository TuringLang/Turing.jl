using Turing
import Turing.translate!

ex = quote
  x = 1
  y = rand()
  y ~ Normal(0,1)
end

res = translate!(:(y~Normal(1,1)))

Base.@assert res.head == :macrocall
Base.@assert res.args[1] == Symbol("@~")
Base.@assert res.args[3] == :y
Base.@assert res.args[4] == :(Normal(1, 1))


res2 = translate!(ex)

Base.@assert res2.args[end].head == :macrocall
Base.@assert res2.args[end].args[1] == Symbol("@~")
Base.@assert res2.args[end].args[3] == :y
Base.@assert res2.args[end].args[4] == :(Normal(0, 1))
