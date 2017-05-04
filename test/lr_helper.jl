# Load data
# NOTE: put the data file in the same path of this file
readlrdata() = begin
  x = Matrix{Float64}(0, 24)
  y = Vector{Float64}()
  cd(dirname(@__FILE__))
  open("lr_nuts.data") do f
    while !eof(f)
      raw_line = readline(f)
      data_str = filter(str -> length(str) > 0, split(raw_line, r"[ ]+")[1:end-1])
      data = map(str -> parse(str), data_str)
      x = cat(1, x, data[1:end-1]')
      y = cat(1, y, data[end])
    end
  end
  x, y
end
