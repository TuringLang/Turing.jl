using Turing

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

API_list = [Prior, PriorArray, PriorContainer, addPrior]

cd(joinpath(dirname(@__FILE__),"source")) do
  for API in API_list
    fname = replace("$(string(API))", "Turing.", "")
    open("$fname.rst","w") do f
      md = Base.doc(API)
      if isa(md,Markdown.MD)
        isa(md.content[1].content[1],Markdown.Code) || error("Incorrect docstring format: $D")

        printrst(f,md)
      else
        warn("$D is not documented.")
      end
    end
  end
end
