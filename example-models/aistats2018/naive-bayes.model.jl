using Turing

@model nb(images, labels, C, D, N, label_mask) = begin
    
    m = Vector{Vector{Real}}(C)
    # @simd for c = 1:C
    #     @inbounds m[c] ~ MvNormal(zeros(D), 10 * ones(D))
    # end

    m ~ [MvNormal(zeros(D), 10 * ones(D))]
    
    # for c = 1:C
    #     @simd for d = 1:D
    #         @inbounds _lp += sum(logpdf.(Normal(m[c][d], 1), images[d, label_mask[c]]))
    #     end
    # end

    @simd for n = 1:N
        @inbounds _lp += logpdf(MvNormal(zeros(D), 10 * ones(D)), images[:,n] - m[labels[n]])
    end
   
    # @simd for c = 1:C
    #     @inbounds _lp += mapreduce(d -> sum(logpdf.(Normal(m[c][d], λ), images[d, label_mask[c]])), +, 1:D)
    # end

    # @simd for n = 1:N
    #     # @inbounds _lp += sum(logpdf.(Normal(m[l][d], λ), images[d, label_mask[l]]))
    #     @inbounds images[:,n] ~ MvNormal(m[labels[n]], ones(D))
    # end

end