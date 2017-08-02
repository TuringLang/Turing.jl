readlrdata() = begin
  N = 2000 #Note: Intial size of dataset is 32562, took the first 2000 points
  d = 123
  x = zeros(Float64, N, d)
  y = zeros(Int32, N)
  i = 1
  cd(dirname(@__FILE__))
  open("a9a2000.data") do f
    while !eof(f)
      raw_line = readline(f)
      data_str = filter(str -> length(str) > 0, split(raw_line, r"[ ]+")[1:end-1])
      data = map(str -> parse(str), data_str)
      x_tmp = zeros(Int32, d)
      x_tmp[data[2:end]] = 1
      x[i, :] = x_tmp'
      y[i] = data[1] - 1
      i += 1
    end
  end
  x, y
end
