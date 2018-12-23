using Turing
import Turing.translate_tilde!

model_info = Dict(:name => "model", :main_body_names => Dict(:model => :model, :vi => :vi, :sampler => :sampler), :arg_syms => [], :tent_pvars_list => [])

ex = :(y ~ Normal(1,1))
model_info[:main_body] = ex
translate_tilde!(model_info)
res = model_info[:main_body]
Base.@assert res.head == :block

ex = quote
  x = 1
  y = rand()
  y ~ Normal(0,1)
end
model_info[:main_body] = ex
translate_tilde!(model_info)
res = model_info[:main_body]
Base.@assert res.head == :block
