module ASCIIPlots

export ASCIIPlot, scatterplot

# Code adapted from: https://github.com/johnmyleswhite/ASCIIPlots.jl/blob/master/src/ASCIIPlots.jl

immutable ASCIIPlot
    s::String
end

Base.show(io::IO, p::ASCIIPlot) = print(io, p.s)

function scatterplot(x::AbstractArray, y::AbstractArray; sym::Char = '^')
    x, y = vec(x), vec(y)

    # Sanity checking
    N = length(x)
    if N != length(y)
        error("x and y must have the same length")
    end

    # Resolution along x and y dimensions
    res_x, res_y = 60, 20

    # Standarize data scale
    minx = minimum(x)
    maxx = maximum(x)
    miny = minimum(y)
    maxy = maximum(y)
    x = x .- minx
    x = x / maximum(x)
    y = y .- miny
    y = y / maximum(y)

    # Snap data points to a grid
    xi = floor(Integer, x * (res_x - 1)) .+ 1
    yi = floor(Integer, y * (res_y - 1)) .+ 1

    # Is there a point at location (i, j)?
    A = zeros(res_y, res_x)
    for i in 1:N
        A[yi[i], xi[i]] = 1
    end

    io = IOBuffer()

    print(io, "\n")

    # Top grid line
    print(io, "\t")
    for j = 1:(res_x + 1)
        print(io, "-")
    end
    print(io, "\n")

    for i = res_y:-1:1
        # Left grid line
        print(io, "\t|")

        # Data points
        for j = 1:res_x
            if A[i, j] == 1
                print(io, sym)
            else
                print(io, " ")
            end
        end

        # Right grid line + Y tick marks
        if i == res_y
            @printf io "| %2.2f\n" maxy
        elseif i == 1
            @printf io "| %2.2f\n" miny
        else
            print(io, "|\n")
        end
    end

    # Bottom grid line
    print(io, "\t")
    for j = 1:(res_x + 1)
        print(io, "-")
    end
    print(io, "\n")

    # Tick marks for X axis
    @printf io "\t%2.2f" minx
    for j = 1:(res_x - 8)
        print(io, " ")
    end
    @printf io "%2.2f" maxx

    print(io, "\n")

    return ASCIIPlot(String(io))
end

function scatterplot(y::AbstractArray; sym::Char = '^')
    scatterplot([1:length(y)], y, sym = sym)
end

end

using ASCIIPlots

# Get running time of Stan
get_stan_time(stan_model_name::String) = begin
  s = readlines(pwd()*"/tmp/$(stan_model_name)_samples_1.csv")
  m = match(r"(?<time>[0-9].[0-9]*)", s[end-1])
  float(m[:time])
end

# Run benchmark
tbenchmark(alg::String, model::String, data::String) = begin
  chain, time, mem, _, _  = eval(parse("@timed sample($model($data), $alg)"))
  alg, time, mem, chain
end

# Build logd from Turing chain
build_logd(name::String, engine::String, time, mem, tchain) = begin
  Dict(
    "name" => name,
    "engine" => engine,
    "time" => time,
    "mem" => mem,
    "turing" => Dict(v => mean(tchain[Symbol(v)]) for v in keys(tchain))
  )
end

# Log function
print_log(logd::Dict, monitor=[]) = begin
  println("/=======================================================================")
  println("| Benchmark Result for >>> $(logd["name"]) <<<")
  println("|-----------------------------------------------------------------------")
  println("| Overview")
  println("|-----------------------------------------------------------------------")
  println("| Inference Engine  : $(logd["engine"])")
  println("| Time Used (s)     : $(logd["time"])")
  if haskey(logd, "time_stan")
    println("|   -> time by Stan : $(logd["time_stan"])")
  end
  println("| Mem Alloc (bytes) : $(logd["mem"])")
  if haskey(logd, "turing")
    println("|-----------------------------------------------------------------------")
    println("| Turing Inference Result")
    println("|-----------------------------------------------------------------------")
    for (v, m) = logd["turing"]
      if isempty(monitor) || v in monitor
        println("| >> $v <<")
        println("| mean = $(round(m, 3))")
        if haskey(logd, "analytic") && haskey(logd["analytic"], v)
          print("|   -> analytic = $(round(logd["analytic"][v], 3)), ")
          diff = abs(m - logd["analytic"][v])
          diff_output = "diff = $(round(diff, 3))"
          if sum(diff) > 0.2
            print_with_color(:red, diff_output*"\n")
          else
            println(diff_output)
          end
        end
        if haskey(logd, "stan") && haskey(logd["stan"], v)
          print("|   -> Stan     = $(round(logd["stan"][v], 3)), ")
          diff = abs(m - logd["stan"][v])
          diff_output = "diff = $(round(diff, 3))"
          if sum(diff) > 0.2
            print_with_color(:red, diff_output*"\n")
          else
            println(diff_output)
          end
        end
      end
    end
  end
  println("\\=======================================================================")
end
