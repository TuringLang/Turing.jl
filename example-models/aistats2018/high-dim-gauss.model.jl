using Turing

@model high_dim_gauss(D) = begin
    
    mu ~ MvNormal(zeros(D), ones(D))

    # mu = Vector{Real}(D)
    # mu ~ [Normal(0, 1)]
    
end