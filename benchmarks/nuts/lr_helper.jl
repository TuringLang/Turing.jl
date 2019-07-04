readlrdata() = begin
    x = Matrix{Float64}(undef, 0, 24)
    y = Vector{Float64}()
    open("lr_nuts.data") do f
        while !eof(f)
            raw_line = readline(f)
            s = split(raw_line, r"[  ]+")[1:end-1]
            data_str = filter(str -> length(str) > 0, s)
            data = map(str -> parse(Float64, str), data_str)
            x = vcat(x, data[1:end-1]')
            y = vcat(y, data[end] - 1)  # turn {1, 2} to {0, 1}
        end
    end
    return x, y
end
