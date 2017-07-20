@model binormal() = begin
  y ~ MvNormal(zeros(2), [1.0 0.1; 0.1 1.0])
end
