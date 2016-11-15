####################
# Helper functions #
####################

if VERSION < v"0.5.0-"
    function Markdown.rstinline(io::IO, md::Markdown.Link)
        if ismatch(r":(func|obj|ref|exc|class|const|data):`\.*", md.url)
            Markdown.rstinline(io, md.url)
        else
            Markdown.rstinline(io, "`", md.text, " <", md.url, ">`_")
        end
    end
end

function printrst(io,md)
    mdd = md.content[1]
    sigs = shift!(mdd.content)

    decl = ".. function:: "*replace(sigs.code, "\n","\n              ")
    body = Markdown.rst(mdd)
    println(io, decl)
    println(io)
    for l in split(body, "\n")
        ismatch(r"^\s*$", l) ? println(io) : println(io, "   ", l)
    end
end

# Load the package to bind docs
using Turing

# NOTE: this is the to-generate list. Each key-value mapping will be convereted into a .rst file. :title is the title of this .rst file and :list contains APIs to be generated.
to_gen = Dict(
  "replayapi" => Dict(
    :title  =>  "Replay",
    :list   =>  ["Prior", "PriorArray", "PriorContainer", "addPrior"]
  ),
  "compilerapi" => Dict(
    :title  =>  "Compiler",
    :list   =>  ["@assume", "@observe", "@predict", "@model"]
  ),
  "samplerapi" => Dict(
    :title  =>  "Sampler",
    :list   =>  ["IS", "SMC", "PG", "HMC"]
  ),
  "tarray" => Dict(
    :title  =>  "TArray",
    :list   =>  ["TArray", "tzeros"]
  )
)

# Generate all APIs
cd(joinpath(dirname(@__FILE__),"source")) do
  for fname in keys(to_gen)
    open("$fname.rst","w") do f
      println(f,"$(to_gen[fname][:title])\n=========\n")
      for api in to_gen[fname][:list]
        md = include_string("@doc $api")
        if isa(md,Markdown.MD)
          isa(md.content[1].content[1],Markdown.Code) || error("Incorrect docstring format: $D")

          printrst(f,md)
        else
          warn("$D is not documented.")
        end
      end
    end
  end
end

# Generate API filenames
fnames = [fname for fname in keys(to_gen)]
api_str = "$(shift!(fnames))"
for fname in fnames
  api_str *= "\n   $fname"
end

# Generate index.rst
rst = """
Welcome to Turing.jl's documentation!
=====================================

Contents
^^^^^^^^

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   getstarted

.. toctree::
   :maxdepth: 2
   :caption: APIs

   $api_str

.. toctree::
   :maxdepth: 2
   :caption: Development Notes

   language
   compiler
   samplerintro
   tarray
   workflow

.. toctree::
   :maxdepth: 2
   :caption: License

   license

"""

cd(joinpath(dirname(@__FILE__),"source")) do
  open("index.rst","w") do f
    println(f,rst)
  end
end
