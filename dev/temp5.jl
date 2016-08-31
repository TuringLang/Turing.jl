using Turing

@model test begin
  @assume v ~ Chisq(10)
  @observe 1.5 ~ Normal(v, 1)
  @predict v
end

sample(test, PG(10, 10))
