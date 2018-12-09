using Turing
import Turing.translate!

model_info = Dict(:name => :model, 
                  :args => [], 
                  :dvars => Set{Symbol}(), 
                  :pvars => Set{Symbol}())
res = translate!(:(y~Normal(1,1)), model_info)
Base.@assert res.head == :block

ex = quote
  x = 1
  y = rand()
  y ~ Normal(0,1)
end
model_info = Dict(:name => :model, 
                  :args => [], 
                  :dvars => Set{Symbol}(), 
                  :pvars => Set{Symbol}())
res2 = translate!(ex, model_info)
Base.@assert res.head == :block
