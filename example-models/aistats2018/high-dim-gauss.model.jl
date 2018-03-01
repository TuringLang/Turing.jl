using Turing

@model high_dim_gauss(D) = begin
    
    mu ~ MvNormal(zeros(D), ones(D))
    
end