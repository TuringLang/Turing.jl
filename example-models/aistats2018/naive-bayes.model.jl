using Turing

@model nb(images, labels, C, D, N, label_mask, λ=1) = begin
    
    m = Vector{Vector{Real}}(C)
    @simd for c = 1:C
        @inbounds m[c] ~ MvNormal(zeros(D), 10 * ones(D))
    end
    
    for d = 1:D, l = 1:C
        @inbounds _lp += sum(logpdf.(Normal(m[l][d], λ), images[d, label_mask[l]]))
    end
    
end