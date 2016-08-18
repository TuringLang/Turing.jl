@model f(data, theta) = begin
  μ ~ Normal(0, 1; paramValue=theta[gensym(),counter])
  σ ~ Gamma(1,1)
  y ~ Normal(μ, sqrt(σ); obsValue=data)
end

@model f(data) = begin
  @assume μ ~ Normal(0, 1; param=true) # IArray (forbid loops or function calls)
  @assume ϕ ~ Inf[Normal(0, 1; param=true)] # IArray (forbid loops or function calls)
  @assume σ ~ Gamma(1,1)
  @observe y ~ Normal(μ, sqrt(σ))
end