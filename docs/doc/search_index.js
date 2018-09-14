var documenterSearchIndex = {"docs": [

{
    "location": "index.html#",
    "page": "Documentation",
    "title": "Documentation",
    "category": "page",
    "text": ""
},

{
    "location": "index.html#Documentation-1",
    "page": "Documentation",
    "title": "Documentation",
    "category": "section",
    "text": "Turing is a universal probabilistic programming language with a focus on an intuitive modelling interface, composable probabilistic inference and computational scalability.Turing provides Hamiltonian Monte Carlo (HMC) and particle MCMC sampling algorithms for complex posterior distributions (e.g. those involving discrete variables and stochastic control flows). Current features include:Universal probabilistic programming with an intuitive modelling interface\nHamiltonian Monte Carlo (HMC) sampling for differentiable posterior distributions\nParticle MCMC sampling for complex posterior distributions involving discrete variables and stochastic control flow\nGibbs sampling that combines particle MCMC,  HMC and many other MCMC algorithms"
},

{
    "location": "index.html#Resources-1",
    "page": "Documentation",
    "title": "Resources",
    "category": "section",
    "text": "Please visit Turing.jl wiki for documentation, tutorials (e.g. get started) and other topics (e.g. advanced usages). Below are some example models for Turing.Introduction\nGaussian Mixture Model\nBayesian Hidden Markov Model\nFactorical Hidden Markov Model\nTopic Models: LDA and MoC"
},

{
    "location": "index.html#Citing-Turing-1",
    "page": "Documentation",
    "title": "Citing Turing",
    "category": "section",
    "text": "To cite Turing, please refer to the following paper. A sample BiBTeX entry entry is given below:@InProceedings{turing17,\n  title = {{T}uring: a language for flexible probabilistic inference},\n  author = {Ge, Hong and Xu, Kai and Ghahramani, Zoubin},\n  booktitle = {Proceedings of the 21th International Conference on Artificial Intelligence and Statistics},\n  year = {2018},\n  series = {Proceedings of Machine Learning Research},\n  publisher = {PMLR},\n}"
},

{
    "location": "index.html#Other-Probablistic/Deep-Learning-Languages-1",
    "page": "Documentation",
    "title": "Other Probablistic/Deep Learning Languages",
    "category": "section",
    "text": "Stan\nInfer.NET\nPyTorch / Pyro\nTensorFlow / Edward\nDyNet"
},

{
    "location": "get-started.html#",
    "page": "Getting Started",
    "title": "Getting Started",
    "category": "page",
    "text": ""
},

{
    "location": "get-started.html#Getting-Started-1",
    "page": "Getting Started",
    "title": "Getting Started",
    "category": "section",
    "text": ""
},

{
    "location": "get-started.html#Installation-1",
    "page": "Getting Started",
    "title": "Installation",
    "category": "section",
    "text": "To use Turing, you need install Julia first and then install Turing."
},

{
    "location": "get-started.html#Install-Julia-1",
    "page": "Getting Started",
    "title": "Install Julia",
    "category": "section",
    "text": "You will need Julia 1.0, which you can get from the official Julia website.There are three options for users:A command line version Julia/downloads (recommended).\nA community maintained IDE Juno.\nJuliaBox.com – a Jupyter notebook in the browser.For the command line version, we recommend that you install a version downloaded from Julia\'s official website, as Turing may not work correctly with Julia provided by other sources (e.g. Turing does not work with Julia installed via apt-get due to missing header files).Juno will also the command line version installed. This IDE is recommended for heavy users who require features like debugging, quick documentation check, etc.JuliaBox provides a pre-installed Jupyter notebook for Julia. You can take a shot at Turing without installing Julia on your machine in few seconds."
},

{
    "location": "get-started.html#Install-Turing.jl-1",
    "page": "Getting Started",
    "title": "Install Turing.jl",
    "category": "section",
    "text": "Turing is an officially registered Julia package, so the following will install a stable version of Turing while inside Julia\'s package manager (press ] from the REPL):add TuringIf you want to use the latest version of Turing with some experimental features, you can try the following instead:add Turing#master\ntest TuringIf all tests pass, you\'re ready to start using Turing."
},

{
    "location": "advanced.html#",
    "page": "Advanced Usage",
    "title": "Advanced Usage",
    "category": "page",
    "text": ""
},

{
    "location": "advanced.html#Advanced-Usage-1",
    "page": "Advanced Usage",
    "title": "Advanced Usage",
    "category": "section",
    "text": ""
},

{
    "location": "advanced.html#How-to-Define-a-Customized-Distribution-1",
    "page": "Advanced Usage",
    "title": "How to Define a Customized Distribution",
    "category": "section",
    "text": "Turing.jl supports the use of distributions from the Distributions.jl package. By extension it also supports the use of customized distributions, by defining them as subtypes of Distribution type of the Distributions.jl package, as well as corresponding functions.Below shows a workflow of how to define a customized distribution, using a flat prior as a simple example."
},

{
    "location": "advanced.html#.-Define-the-Distribution-Type-1",
    "page": "Advanced Usage",
    "title": "1. Define the Distribution Type",
    "category": "section",
    "text": "First, define a type of the distribution, as a subtype of a corresponding distribution type in the Distributions.jl package.immutable Flat <: ContinuousUnivariateDistribution\nend"
},

{
    "location": "advanced.html#.-Implement-Sampling-and-Evaluation-of-the-log-pdf-1",
    "page": "Advanced Usage",
    "title": "2. Implement Sampling and Evaluation of the log-pdf",
    "category": "section",
    "text": "Second, define rand() and logpdf(), which will be used to run the model.Distributions.rand(d::Flat) = rand()\nDistributions.logpdf{T<:Real}(d::Flat, x::T) = zero(x)"
},

{
    "location": "advanced.html#.-Define-Helper-Functions-1",
    "page": "Advanced Usage",
    "title": "3. Define Helper Functions",
    "category": "section",
    "text": "In most cases, it may be required to define helper functions, such as the minimum, maximum, rand, and logpdf functions, among others."
},

{
    "location": "advanced.html#.1-Domain-Transformation-1",
    "page": "Advanced Usage",
    "title": "3.1 Domain Transformation",
    "category": "section",
    "text": "Some helper functions are necessary for domain transformation. For univariate distributions, the necessary ones to implement are minimum() and maximum().Distributions.minimum(d::Flat) = -Inf\nDistributions.maximum(d::Flat) = +InfFunctions for domain transformation which may be required by multivariate or matrix-variate distributions are size(d), link(d, x) and invlink(d, x). Please see Turing\'s transform.jl for examples."
},

{
    "location": "advanced.html#.2-Vectorization-Support-1",
    "page": "Advanced Usage",
    "title": "3.2 Vectorization Support",
    "category": "section",
    "text": "The vectorization syntax follows rv ~ [distribution], which requires rand() and logpdf() to be called on multiple data points at once. An appropriate implementation for Flat are shown below.Distributions.rand(d::Flat, n::Int) = Vector([rand() for _ = 1:n])\nDistributions.logpdf{T<:Real}(d::Flat, x::Vector{T}) = zero(x)"
},

{
    "location": "advanced.html#Avoid-Using-the-@model-Macro-1",
    "page": "Advanced Usage",
    "title": "Avoid Using the @model Macro",
    "category": "section",
    "text": "When integrating Turing.jl with other libraries, it\'s can be necessary to avoid using the @model macro. To achieve this, one needs to understand the @model macro, which works as a closure and generates an amended function byassigning the arguments to corresponding local variables;\nadding two keyword arguments vi=VarInfo() and sampler=nothing to the scope; and\nforcing the function to return vi.Thus by doing these three steps manually, one can get rid of the @model macro. Taking the gdemo model as an example, the two code sections below (macro and macro-free) are equivalent.@model gdemo(x) = begin\n    s ~ InverseGamma(2,3)\n    m ~ Normal(0,sqrt(s))\n    x[1] ~ Normal(m, sqrt(s))\n    x[2] ~ Normal(m, sqrt(s))\n    return s, m\nend\n\nmf = gdemo([1.5, 2.0])\nsample(mf, HMC(1000, 0.1, 5))# Force Turing.jl to initialize its compiler\nmf(vi, sampler; x=[1.5, 2.0]) = begin\n  s = Turing.assume(sampler,\n                    InverseGamma(2, 3),\n                    Turing.VarName(vi, [:c_s, :s], \"\"),\n                    vi)\n  m = Turing.assume(sampler,\n                    Normal(0,sqrt(s)),\n                    Turing.VarName(vi, [:c_m, :m], \"\"),\n                    vi)\n  for i = 1:2\n    Turing.observe(sampler,\n                   Normal(m, sqrt(s)),\n                   x[i],\n                   vi)\n  end\n  vi\nend\nmf() = mf(Turing.VarInfo(), nothing)\n\nsample(mf, HMC(1000, 0.1, 5))Note that the use of ~ must be removed due to the fact that in Julia 0.6, ~ is no longer a macro. For this reason, Turing.jl parses ~ within the @model macro to allow for this intuitive notation."
},

{
    "location": "advanced.html#Task-Copying-1",
    "page": "Advanced Usage",
    "title": "Task Copying",
    "category": "section",
    "text": "Turing copies Julia tasks to deliver efficient inference algorithms, but it also provides alternative slower implementation as a fallback. Task copying is enabled by default. Task copying requires building a small C program, which should be done automatically on Linux and Mac systems that have GCC and Make installed."
},

{
    "location": "contributing/guide.html#",
    "page": "Contributing",
    "title": "Contributing",
    "category": "page",
    "text": ""
},

{
    "location": "contributing/guide.html#Contributing-1",
    "page": "Contributing",
    "title": "Contributing",
    "category": "section",
    "text": "Turing is an open source project. If you feel that you have some relevant skills and are interested in contributing, then please do get in touch. You can contribute by opening issues on GitHub or implementing things yourself and making a pull request. We would also appreciate example models written using Turing.Turing has a style guide. It is not strictly necessary to review it before making a pull request, but you may be asked to change portions of your code to conform with the style guide before it is merged."
},

{
    "location": "contributing/guide.html#How-to-Contribute-1",
    "page": "Contributing",
    "title": "How to Contribute",
    "category": "section",
    "text": ""
},

{
    "location": "contributing/guide.html#Getting-started-1",
    "page": "Contributing",
    "title": "Getting started",
    "category": "section",
    "text": "Fork this repository.\nClone your fork on your local machine: git clone https://github.com/your_username/Turing.jl.\nAdd a remote corresponding to this repository:git remote add upstream https://github.com/TuringLang/Turing.jl."
},

{
    "location": "contributing/guide.html#What-can-I-do-?-1",
    "page": "Contributing",
    "title": "What can I do ?",
    "category": "section",
    "text": "Look at the issues page to find an outstanding issue. For instance, you could implement new features, fix bugs or write example models."
},

{
    "location": "contributing/guide.html#Git-workflow-1",
    "page": "Contributing",
    "title": "Git workflow",
    "category": "section",
    "text": "Make sure that your local master branch is up to date with this repository\'s one (for more details):git fetch upstream\ngit checkout master\ngit rebase upstream/masterCreate a new branch: git checkout -b branch_name (usually use feature-issue_id or bugfix-issue_id).\nDo your stuff: git add ..., git commit -m \'...\'.\nPush your local branch to your fork of this repository: git push --set-upstream origin branch_name."
},

{
    "location": "contributing/guide.html#Make-a-pull-request-1",
    "page": "Contributing",
    "title": "Make a pull request",
    "category": "section",
    "text": "Create a pull request by going to this repository front page and selecting Compare & pull request.\nIf related to a specific issue, link the pull request link in that issue, and in the pull request also link the issue."
},

{
    "location": "contributing/style_guide.html#",
    "page": "Style Guide",
    "title": "Style Guide",
    "category": "page",
    "text": ""
},

{
    "location": "contributing/style_guide.html#Style-Guide-1",
    "page": "Style Guide",
    "title": "Style Guide",
    "category": "section",
    "text": "This style guide is adapted from Invenia\'s style guide. We would like to thank them for allowing us to access and use it. Please don\'t let not having read it stop you from contributing to Turing! No one will be annoyed if you open a PR whose style doesn\'t follow these conventions; we will just help you correct it before it gets merged.These conventions were originally written at Invenia, taking inspiration from a variety of sources including Python\'s PEP8, Julia\'s Notes for Contributors, and Julia\'s Style Guide.What follows is a mixture of a verbatim copy of Invenia\'s original guide and some of our own modifications."
},

{
    "location": "contributing/style_guide.html#A-Word-on-Consistency-1",
    "page": "Style Guide",
    "title": "A Word on Consistency",
    "category": "section",
    "text": "When adhering to this style it\'s important to realize that these are guidelines and not rules. This is stated best in the PEP8:A style guide is about consistency. Consistency with this style guide is important. Consistency within a project is more important. Consistency within one module or function is most important.But most importantly: know when to be inconsistent – sometimes the style guide just doesn\'t apply. When in doubt, use your best judgment. Look at other examples and decide what looks best. And don\'t hesitate to ask!"
},

{
    "location": "contributing/style_guide.html#Synopsis-1",
    "page": "Style Guide",
    "title": "Synopsis",
    "category": "section",
    "text": "Attempt to follow both the Julia Contribution Guidelines, the Julia Style Guide, and this guide. When convention guidelines conflict this guide takes precedence (known conflicts will be noted in this guide).Use 4 spaces per indentation level, no tabs.\nTry to adhere to a 92 character line length limit.\nUse upper camel case convention for modules and types.\nUse lower case with underscores for method names (note: Julia code likes to use lower case without underscores).\nComments are good, try to explain the intentions of the code.\nUse whitespace to make the code more readable.\nNo whitespace at the end of a line (trailing whitespace).\nAvoid padding brackets with spaces. ex. Int64(value) preferred over Int64( value )."
},

{
    "location": "contributing/style_guide.html#Editor-Configuration-1",
    "page": "Style Guide",
    "title": "Editor Configuration",
    "category": "section",
    "text": ""
},

{
    "location": "contributing/style_guide.html#Sublime-Text-Settings-1",
    "page": "Style Guide",
    "title": "Sublime Text Settings",
    "category": "section",
    "text": "If you are a user of Sublime Text we recommend that you have the following options in your Julia syntax specific settings. To modify these settings first open any Julia file (*.jl) in Sublime Text. Then navigate to: Preferences > Settings - More > Syntax Specific - User{\n    \"translate_tabs_to_spaces\": true,\n    \"tab_size\": 4,\n    \"trim_trailing_white_space_on_save\": true,\n    \"ensure_newline_at_eof_on_save\": true,\n    \"rulers\": [92]\n}"
},

{
    "location": "contributing/style_guide.html#Vim-Settings-1",
    "page": "Style Guide",
    "title": "Vim Settings",
    "category": "section",
    "text": "If you are a user of Vim we recommend that you add the following options to your .vimrc file.set tabstop=4                             \" Sets tabstops to a width of four columns.\nset softtabstop=4                         \" Determines the behaviour of TAB and BACKSPACE keys with expandtab.\nset shiftwidth=4                          \" Determines the results of >>, <<, and ==.\n\nau FileType julia setlocal expandtab      \" Replaces tabs with spaces.\nau FileType julia setlocal colorcolumn=93 \" Highlights column 93 to help maintain the 92 character line limit.By default, Vim seems to guess that .jl files are written in Lisp. To ensure that Vim recognizes Julia files you can manually have it check for the .jl extension, but a better solution is to install Julia-Vim, which also includes proper syntax highlighting and a few cool other features."
},

{
    "location": "contributing/style_guide.html#Atom-Settings-1",
    "page": "Style Guide",
    "title": "Atom Settings",
    "category": "section",
    "text": "Atom defaults preferred line length to 80 characters. We want that at 92 for julia. To change it:Go to Atom -> Preferences -> Packages.\nSearch for the \"language-julia\" package and open the settings for it.\nFind preferred line length (under \"Julia Grammar\") and change it to 92."
},

{
    "location": "contributing/style_guide.html#Code-Formatting-1",
    "page": "Style Guide",
    "title": "Code Formatting",
    "category": "section",
    "text": ""
},

{
    "location": "contributing/style_guide.html#Function-Naming-1",
    "page": "Style Guide",
    "title": "Function Naming",
    "category": "section",
    "text": "Names of functions should describe an action or property irrespective of the type of the argument; the argument\'s type provides this information instead. For example, buyfood(food) should be buy(food::Food).Names of functions should usually be limited to one or two lowercase words. Ideally write buyfood not buy_food, but if you are writing a function whose name is hard to read without underscores then please do use them."
},

{
    "location": "contributing/style_guide.html#Method-Definitions-1",
    "page": "Style Guide",
    "title": "Method Definitions",
    "category": "section",
    "text": "Only use short-form function definitions when they fit on a single line:# Yes:\nfoo(x::Int64) = abs(x) + 3\n# No:\nfoobar(array_data::AbstractArray{T}, item::T) where {T<:Int64} = T[\n    abs(x) * abs(item) + 3 for x in array_data\n]\n\n# No:\nfoobar(\n    array_data::AbstractArray{T},\n    item::T,\n) where {T<:Int64} = T[abs(x) * abs(item) + 3 for x in array_data]\n# Yes:\nfunction foobar(array_data::AbstractArray{T}, item::T) where T<:Int64\n    return T[abs(x) * abs(item) + 3 for x in array_data]\nendWhen using long-form functions always use the return keyword:# Yes:\nfunction fnc(x::T) where T\n    result = zero(T)\n    result += fna(x)\n    return result\nend\n# No:\nfunction fnc(x::T) where T\n    result = zero(T)\n    result += fna(x)\nend\n\n# Yes:\nfunction Foo(x, y)\n    return new(x, y)\nend\n# No:\nfunction Foo(x, y)\n    new(x, y)\nendFunctions definitions with parameter lines which exceed 92 characters should separate each parameter by a newline and indent by one-level:# Yes:\nfunction foobar(\n    df::DataFrame,\n    id::Symbol,\n    variable::Symbol,\n    value::AbstractString,\n    prefix::AbstractString=\"\",\n)\n    # code\nend\n\n# Ok:\nfunction foobar(df::DataFrame, id::Symbol, variable::Symbol, value::AbstractString, prefix::AbstractString=\"\")\n    # code\nend\n# No:\nfunction foobar(df::DataFrame, id::Symbol, variable::Symbol, value::AbstractString,\n    prefix::AbstractString=\"\")\n\n    # code\nend\n# No:\nfunction foobar(\n        df::DataFrame,\n        id::Symbol,\n        variable::Symbol,\n        value::AbstractString,\n        prefix::AbstractString=\"\",\n    )\n    # code\nend"
},

{
    "location": "contributing/style_guide.html#Keyword-Arguments-1",
    "page": "Style Guide",
    "title": "Keyword Arguments",
    "category": "section",
    "text": "When calling a function always separate your keyword arguments from your positional arguments with a semicolon. This avoids mistakes in ambiguous cases (such as splatting a Dict).# Yes:\nxy = foo(x; y=3)\n# No:\nxy = foo(x, y=3)"
},

{
    "location": "contributing/style_guide.html#Whitespace-1",
    "page": "Style Guide",
    "title": "Whitespace",
    "category": "section",
    "text": "Avoid extraneous whitespace in the following situations:Immediately inside parentheses, square brackets or braces.\njulia   Yes: spam(ham[1], [eggs])   No:  spam( ham[ 1 ], [ eggs ] )\nImmediately before a comma or semicolon:\njulia   Yes: if x == 4 @show(x, y); x, y = y, x end   No:  if x == 4 @show(x , y) ; x , y = y , x end\nWhen using ranges unless additional operators are used:\njulia   Yes: ham[1:9], ham[1:3:9], ham[1:3:end]   No:  ham[1: 9], ham[1 : 3: 9]\njulia   Yes: ham[lower:upper], ham[lower:step:upper]   Yes: ham[lower + offset : upper + offset]   Yes: ham[(lower + offset):(upper + offset)]   No:  ham[lower + offset:upper + offset]\nMore than one space around an assignment (or other) operator to align it with another:\n```\nYes:\nx = 1   y = 2   long_variable = 3\nNo:\nx             = 1   y             = 2   long_variable = 3   ```\nAlways surround these binary operators with a single space on either side: assignment (=), updating operators (+=, -=, etc.), numeric comparisons operators (==, <, >, !=, etc.). Note that this guideline does not apply when performing assignment in method definitions.\n```   Yes: i = i + 1   No:  i=i+1\nYes: submitted += 1   No:  submitted +=1\nYes: x^2 < y   No:  x^2<y   ```\nAssignments using expanded array, tuple, or function notation should have the first open bracket on the same line assignment operator and the closing bracket should match the indentation level of the assignment. Alternatively you can perform assignments on a single line when they are short:\n```julia\nYes:\narr = [       1,       2,       3,   ]   arr = [       1, 2, 3,   ]   result = Function(       arg1,       arg2,   )   arr = [1, 2, 3]# No:\narr =\n[\n    1,\n    2,\n    3,\n]\narr =\n[\n    1, 2, 3,\n]\narr = [\n    1,\n    2,\n    3,\n    ]\n```Nested array or tuples that are in expanded notation should have the opening and closing brackets at the same indentation level:\n```julia\nYes:\nx = [       [           1, 2, 3,       ],       [           \"hello\",           \"world\",       ],       [\'a\', \'b\', \'c\'],   ]\nNo:\ny = [       [           1, 2, 3,       ], [           \"hello\",           \"world\",       ],   ]   z = [[           1, 2, 3,       ], [           \"hello\",           \"world\",       ],   ]   ```\nAlways include the trailing comma when working with expanded arrays, tuples or functions notation. This allows future edits to easily move elements around or add additional elements. The trailing comma should be excluded when the notation is only on a single-line:\n```julia\nYes:\narr = [       1,       2,       3,   ]   result = Function(       arg1,       arg2,   )   arr = [1, 2, 3]\nNo:\narr = [       1,       2,       3   ]   result = Function(       arg1,       arg2   )   arr = [1, 2, 3,]   ```\nTriple-quotes use the indentation of the lowest indented line (excluding the opening triple-quote). This means the closing triple-quote should be aligned to least indented line in the string. Triple-backticks should also follow this style even though the indentation does not matter for them.\n```julia\nYes:\nstr = \"\"\"       hello       world!       \"\"\"   str = \"\"\"           hello       world!       \"\"\"   cmd = program           --flag value           parameter\nNo:\nstr = \"\"\"       hello       world!   \"\"\"   ```"
},

{
    "location": "contributing/style_guide.html#Comments-1",
    "page": "Style Guide",
    "title": "Comments",
    "category": "section",
    "text": "Comments should be used to state the intended behaviour of code. This is especially important when the code is doing something clever that may not be obvious upon first inspection. Avoid writing comments that state exactly what the code obviously does.# Yes:\nx = x + 1      # Compensate for border\n\n# No:\nx = x + 1      # Increment xComments that contradict the code are much worse than no comments. Always make a priority of keeping the comments up-to-date with code changes!Comments should be complete sentences. If a comment is a phrase or sentence, its first word should be capitalized, unless it is an identifier that begins with a lower case letter (never alter the case of identifiers!).If a comment is short, the period at the end can be omitted. Block comments generally consist of one or more paragraphs built out of complete sentences, and each sentence should end in a period.Comments should be separated by at least two spaces from the expression and have a single space after the #.When referencing Julia in documentation note that \"Julia\" refers to the programming language while \"julia\" (typically in backticks, e.g. julia) refers to the executable.\n# A commment\ncode\n\n# anothher comment\nmore code\n\nTODO"
},

{
    "location": "contributing/style_guide.html#Documentation-1",
    "page": "Style Guide",
    "title": "Documentation",
    "category": "section",
    "text": "It is recommended that most modules, types and functions should have docstrings. That being said, only exported functions are required to be documented. Avoid documenting methods like == as the built in docstring for the function already covers the details well. Try to document a function and not individual methods where possible as typically all methods will have similar docstrings. If you are adding a method to a function which was defined in Base or another package only add a docstring if the behaviour of your function deviates from the existing docstring.Docstrings are written in Markdown and should be concise. Docstring lines should be wrapped at 92 characters.\"\"\"\n    bar(x[, y])\n\nCompute the Bar index between `x` and `y`. If `y` is missing, compute the Bar index between\nall pairs of columns of `x`.\n\"\"\"\nfunction bar(x, y) ...When types or methods have lots of parameters it may not be feasible to write a concise docstring. In these cases it is recommended you use the templates below. Note if a section doesn\'t apply or is overly verbose (for example \"Throws\" if your function doesn\'t throw an exception) it can be excluded. It is recommended that you have a blank line between the headings and the content when the content is of sufficient length. Try to be consistent within a docstring whether you use this additional whitespace. Note that the additional space is only for reading raw markdown and does not effect the rendered version.Type Template (should be skipped if is redundant with the constructor(s) docstring):\"\"\"\n    MyArray{T,N}\n\nMy super awesome array wrapper!\n\n# Fields\n- `data::AbstractArray{T,N}`: stores the array being wrapped\n- `metadata::Dict`: stores metadata about the array\n\"\"\"\nstruct MyArray{T,N} <: AbstractArray{T,N}\n    data::AbstractArray{T,N}\n    metadata::Dict\nendFunction Template (only required for exported functions):\"\"\"\n    mysearch(array::MyArray{T}, val::T; verbose=true) where {T} -> Int\n\nSearches the `array` for the `val`. For some reason we don\'t want to use Julia\'s\nbuiltin search :)\n\n# Arguments\n- `array::MyArray{T}`: the array to search\n- `val::T`: the value to search for\n\n# Keywords\n- `verbose::Bool=true`: print out progress details\n\n# Returns\n- `Int`: the index where `val` is located in the `array`\n\n# Throws\n- `NotFoundError`: I guess we could throw an error if `val` isn\'t found.\n\"\"\"\nfunction mysearch(array::AbstractArray{T}, val::T) where T\n    ...\nendIf your method contains lots of arguments or keywords you may want to exclude them from the method signature on the first line and instead use args... and/or kwargs....\"\"\"\n    Manager(args...; kwargs...) -> Manager\n\nA cluster manager which spawns workers.\n\n# Arguments\n\n- `min_workers::Integer`: The minimum number of workers to spawn or an exception is thrown\n- `max_workers::Integer`: The requested number of worker to spawn\n\n# Keywords\n\n- `definition::AbstractString`: Name of the job definition to use. Defaults to the\n    definition used within the current instance.\n- `name::AbstractString`: ...\n- `queue::AbstractString`: ...\n\"\"\"\nfunction Manager(...)\n    ...\nendFeel free to document multiple methods for a function within the same docstring. Be careful to only do this for functions you have defined.\"\"\"\n    Manager(max_workers; kwargs...)\n    Manager(min_workers:max_workers; kwargs...)\n    Manager(min_workers, max_workers; kwargs...)\n\nA cluster manager which spawns workers.\n\n# Arguments\n\n- `min_workers::Int`: The minimum number of workers to spawn or an exception is thrown\n- `max_workers::Int`: The number of requested workers to spawn\n\n# Keywords\n\n- `definition::AbstractString`: Name of the job definition to use. Defaults to the\n    definition used within the current instance.\n- `name::AbstractString`: ...\n- `queue::AbstractString`: ...\n\"\"\"\nfunction Manager end\nIf the documentation for bullet-point exceeds 92 characters the line should be wrapped and slightly indented. Avoid aligning the text to the :.\"\"\"\n...\n\n# Keywords\n- `definition::AbstractString`: Name of the job definition to use. Defaults to the\n    definition used within the current instance.\n\"\"\"For additional details on documenting in Julia see the official documentation."
},

{
    "location": "contributing/style_guide.html#Test-Formatting-1",
    "page": "Style Guide",
    "title": "Test Formatting",
    "category": "section",
    "text": ""
},

{
    "location": "contributing/style_guide.html#Testsets-1",
    "page": "Style Guide",
    "title": "Testsets",
    "category": "section",
    "text": "Julia provides test sets which allows developers to group tests into logical groupings. Test sets can be nested and ideally packages should only have a single \"root\" test set. It is recommended that the \"runtests.jl\" file contains the root test set which contains the remainder of the tests:@testset \"PkgExtreme\" begin\n    include(\"arithmetic.jl\")\n    include(\"utils.jl\")\nend"
},

{
    "location": "contributing/style_guide.html#Comparisons-1",
    "page": "Style Guide",
    "title": "Comparisons",
    "category": "section",
    "text": "Most tests are written in the form @test x == y. Since the == function doesn\'t take types into account tests like the following are valid: @test 1.0 == 1. Avoid adding visual noise into test comparisons:# Yes:\n@test value == 0\n\n# No:\n@test value == 0.0"
},

{
    "location": "ex/0_Introduction.html#",
    "page": "Introduction to Turing",
    "title": "Introduction to Turing",
    "category": "page",
    "text": ""
},

{
    "location": "ex/0_Introduction.html#Introduction-to-Turing-1",
    "page": "Introduction to Turing",
    "title": "Introduction to Turing",
    "category": "section",
    "text": ""
},

{
    "location": "ex/0_Introduction.html#Introduction-1",
    "page": "Introduction to Turing",
    "title": "Introduction",
    "category": "section",
    "text": "This is the first of a series of tutorials on the universal probabilistic programming language Turing.Turing is probabilistic programming system written entirely in Julia. It has an intuitive modelling syntax and supports a wide range of sampling-based inference algorithms. Most importantly, Turing inference is composable: it combines Markov chain sampling operations on subsets of model variables, e.g. using a combination of a Hamiltonian Monte Carlo (HMC) engine and a particle Gibbs (PG) engine. This composable inference engine allows the user to easily switch between black-box style inference methods such as HMC and customized inference methods.Familiarity with Julia is assumed through out this tutorial. If you are new to Julia, Learning Julia is a good starting point.For users new to Bayesian machine learning, please consider more thorough introductions to the field, such as Pattern Recognition and Machine Learning. This tutorial tries to provide an intuition for Bayesian inference and gives a simple example on how to use Turing. Note that this is not a comprehensive introduction to Bayesian machine learning."
},

{
    "location": "ex/0_Introduction.html#Coin-Flipping-Without-Turing-1",
    "page": "Introduction to Turing",
    "title": "Coin-Flipping Without Turing",
    "category": "section",
    "text": "The following example illustrates the effect of updating our beliefs with every piece of new evidence we observe. In particular, assume that we are unsure about the probability of heads in a coin flip. To get an intuitive understanding of what \"updating our beliefs\" is, we will visualize the probability of heads in a coin flip after each observed evidence.First, let\'s load some of the packages we need to flip a coin (Random, Distributions) and show our results (Plots). You will note that Turing is not an import here — we do not need it for this example. If you are already familiar with posterior updates, you can proceed to the next step.# Using Base modules.\nusing Random\n\n# Load a plotting library.\nusing Plots\n\n# Load the distributions library.\nusing DistributionsNext, we configure our posterior update model. First, let\'s set the true probability that any coin flip will turn up heads and set the number of coin flips we will show our model:# Set the true probability of heads in a coin.\np_true = 0.5\n\n# Iterate from having seen 0 observations to 100 observations.\nNs = 0:100;We will now use the Bernoulli distribution to flip 100 coins, and collect the results in a variable called data:# Draw data from a Bernoulli distribution, i.e. draw heads or tails.\nRandom.seed!(12)\ndata = rand(Bernoulli(p_true), last(Ns))\n\n# Here\'s what the first five coin flips look like:\ndata[1:5]5-element Array{Int64,1}:\n 1\n 0\n 1\n 1\n 0After flipping all our coins, we want to set a prior belief about what we think the distribution of coin flips look like. In this case, we are going to choose a common prior distribution called the Beta distribution.# Our prior belief about the probability of heads in a coin toss.\nprior_belief = Beta(1, 1);With our priors set and our data at hand, we can perform Bayesian inference.This is a fairly simple process. We expose one additional coin flip to our model every iteration, such that the first run only sees the first coin flip, while the last iteration sees all the coin flips. Then, we set the updated_belief variable to an updated version of the original Beta distribution that accounts for the new proportion of heads and tails. For the mathematically inclined, the Beta distribution is updated by adding each coin flip to the distribution\'s alpha and beta parameters, which are initially defined as alpha = 1 beta = 1. Over time, with more and more coin flips, alpha and beta will be approximately equal to each other as we are equally likely to flip a heads or a tails, and the plot of the beta distribution will become more tightly centered around 0.5. This works because mean of the Beta distribution is defined as the following:$$ \\text{E}[\\text{Beta}] = \\dfrac{\\alpha}{\\alpha+\\beta} $$Which is 0.5 when alpha = beta, as we expect for a large enough number of coin flips. As we increase the number of samples, our variance will also decrease, such that the distribution will reflect less uncertainty about the probability of receiving a heads. The definition of the variance for the Beta distribution is the following:$$ \\text{var}[\\text{Beta}] = \\dfrac{\\alpha\\beta}{(\\alpha + \\beta)^2 (\\alpha + \\beta + 1)} $$The intuition about this definition is that the variance of the distribution will approach 0 with more and more samples, as the denominator will grow faster than will the numerator. More samples means less variance.# This is required for plotting only.\nx = range(0, stop = 1, length = 100)\n\n# Make an animation.\nanimation = @animate for (i, N) in enumerate(Ns)\n\n    # Count the number of heads and tails.\n    heads = sum(data[1:i-1])\n    tails = N - heads\n    \n    # Update our prior belief in closed form (this is possible because we use a conjugate prior).\n    updated_belief = Beta(prior_belief.α + heads, prior_belief.β + tails)\n\n    # Plotting\n    plot(x, pdf.(Ref(updated_belief), x), \n        size = (500, 250), \n        title = \"Updated belief after $N observations\",\n        xlabel = \"probability of heads\", \n        ylabel = \"\", \n        legend = nothing,\n        xlim = (0,1),\n        fill=0, α=0.3, w=3)\n    vline!([p_true])\nend;(Image: animation)The animation above shows that with increasing evidence our belief about the probability of heads in a coin flip slowly adjusts towards the true value. The orange line in the animation represents the true probability of seeing heads on a single coin flip, while the mode of the distribution shows what the model believes the probability of a heads is given the evidence it has seen."
},

{
    "location": "ex/0_Introduction.html#Coin-Flipping-With-Turing-1",
    "page": "Introduction to Turing",
    "title": "Coin Flipping With Turing",
    "category": "section",
    "text": "In the previous example, we used the fact that our prior distribution is a conjugate prior. Note that a closed-form expression (the updated_belief expression) for the posterior is not accessible in general and usually does not exist for more interesting models. We are now going to move away from the closed-form expression above and specify the same model using Turing. To do so, we will first need to import Turing, MCMCChain, Distributions, and StatPlots. MCMChain is a library built by the Turing team to help summarize Markov Chain Monte Carlo (MCMC) simulations, as well as a variety of utility functions for diagnostics and visualizations.# Load Turing and MCMCChain.\nusing Turing, MCMCChain\n\n# Load the distributions library.\nusing Distributions\n\n# Load stats plots for density plots.\nusing StatPlotsFirst, we define the coin-flip model using Turing.@model coinflip(y) = begin\n    \n    # Our prior belief about the probability of heads in a coin.\n    p ~ Beta(1, 1)\n    \n    # The number of observations.\n    N = length(y)\n    for n in 1:N\n        # Heads or tails of a coin are drawn from a Bernoulli distribution.\n        y[n] ~ Bernoulli(p)\n    end\nend;After defining the model, we can approximate the posterior distribution by drawing samples from the distribution. In this example, we use a Hamiltonian Monte Carlo sampler to draw these samples. Later tutorials will give more information on the samplers available in Turing and discuss their use for different models.# Settings of the Hamiltonian Monte Carlo (HMC) sampler.\niterations = 1000\nϵ = 0.05\nτ = 10\n\n# Start sampling.\nchain = sample(coinflip(data), HMC(iterations, ϵ, τ));[HMC] Finished with\n  Running time        = 3.9648889480000014;\n  Accept rate         = 0.997;\n  #lf / sample        = 9.99;\n  #evals / sample     = 12.985;\n  pre-cond. diag mat  = [1.0].After finishing the sampling process, we can visualize the posterior distribution approximated using Turing against the posterior distribution in closed-form. We can extract the chain data from the sampler using the Chains(chain[:p]) function, exported from the MCMCChain module. Chains(chain[:p]) creates an instance of the Chain type which summarizes the MCMC simulation — the MCMCChain module supports numerous tools for plotting, summarizing, and describing variables of type Chain.# Construct summary of the sampling process for the parameter p, i.e. the probability of heads in a coin.\np_summary = Chains(chain[:p])\nhistogramplot(p_summary){{< figure src=\"../figures/0Introduction9_1.svg\"  >}}Now we can build our plot:# Compute the posterior distribution in closed-form.\nN = length(data)\nheads = sum(data)\nupdated_belief = Beta(prior_belief.α + heads, prior_belief.β + N - heads)\n\n# Visualize a blue density plot of the approximate posterior distribution using HMC (see Chain 1 in the legend).\np = densityplot(p_summary, xlim = (0,1), legend = :best, w = 2, c = :blue)\n\n# Visualize a green density plot of posterior distribution in closed-form.\nplot!(p, range(0, stop = 1, length = 100), pdf.(Ref(updated_belief), range(0, stop = 1, length = 100)), \n        xlabel = \"probability of heads\", ylabel = \"\", title = \"\", xlim = (0,1), label = \"Closed-form\",\n        fill=0, α=0.3, w=3, c = :lightgreen)\n\n# Visualize the true probability of heads in red.\nvline!(p, [p_true], label = \"True probability\", c = :red);(Image: sdf)As we can see, the Turing model closely approximates the true probability. Hopefully this tutorial has provided an easy-to-follow, yet informative introduction to Turing\'s simpler applications. More advanced usage will be demonstrated in later tutorials."
},

{
    "location": "api.html#",
    "page": "API",
    "title": "API",
    "category": "page",
    "text": ""
},

{
    "location": "api.html#Function-Documentation-1",
    "page": "API",
    "title": "Function Documentation",
    "category": "section",
    "text": "CurrentModule = TuringPages = [\"functions.md\"]\nDepth = 5"
},

{
    "location": "api.html#Turing.@model",
    "page": "API",
    "title": "Turing.@model",
    "category": "macro",
    "text": "@model(name, fbody)\n\nWrapper for models.\n\nUsage:\n\n@model model() = begin\n  # body\nend\n\nExample:\n\n@model gauss() = begin\n  s ~ InverseGamma(2,3)\n  m ~ Normal(0,sqrt.(s))\n  1.5 ~ Normal(m, sqrt.(s))\n  2.0 ~ Normal(m, sqrt.(s))\n  return(s, m)\nend\n\n\n\n\n\n"
},

{
    "location": "api.html#Turing.@~",
    "page": "API",
    "title": "Turing.@~",
    "category": "macro",
    "text": "var_name ~ Distribution()\n\nTilde notation ~ can be used to specifiy a variable follows a distributions.\n\nIf var_name is an un-defined variable or a container (e.g. Vector or Matrix), this variable will be treated as model parameter; otherwise if var_name is defined, this variable will be treated as data.\n\n\n\n\n\n"
},

{
    "location": "api.html#Modelling-1",
    "page": "API",
    "title": "Modelling",
    "category": "section",
    "text": "@model\n@~"
},

{
    "location": "api.html#Turing.Sampler",
    "page": "API",
    "title": "Turing.Sampler",
    "category": "type",
    "text": "Sampler{T}\n\nGeneric interface for implementing inference algorithms. An implementation of an algorithm should include the following:\n\nA type specifying the algorithm and its parameters, derived from InferenceAlgorithm\nA method of sample function that produces results of inference, which is where actual inference happens.\n\nTuring translates models to chunks that call the modelling functions at specified points. The dispatch is based on the value of a sampler variable. To include a new inference algorithm implements the requirements mentioned above in a separate file, then include that file at the end of this one.\n\n\n\n\n\n"
},

{
    "location": "api.html#Turing.Gibbs",
    "page": "API",
    "title": "Turing.Gibbs",
    "category": "type",
    "text": "Gibbs(n_iters, alg_1, alg_2)\n\nCompositional MCMC interface.\n\nUsage:\n\nalg = Gibbs(1000, HMC(1, 0.2, 3, :v1), PG(20, 1, :v2))\n\n\n\n\n\n"
},

{
    "location": "api.html#Turing.HMC",
    "page": "API",
    "title": "Turing.HMC",
    "category": "type",
    "text": "HMC(n_iters::Int, epsilon::Float64, tau::Int)\n\nHamiltonian Monte Carlo sampler.\n\nUsage:\n\nHMC(1000, 0.05, 10)\n\nExample:\n\n# Define a simple Normal model with unknown mean and variance.\n@model gdemo(x) = begin\n  s ~ InverseGamma(2,3)\n  m ~ Normal(0,sqrt.(s))\n  x[1] ~ Normal(m, sqrt.(s))\n  x[2] ~ Normal(m, sqrt.(s))\n  return s, m\nend\n\nsample(gdemo([1.5, 2]), HMC(1000, 0.05, 10))\n\n\n\n\n\n"
},

{
    "location": "api.html#Turing.HMCDA",
    "page": "API",
    "title": "Turing.HMCDA",
    "category": "type",
    "text": "HMCDA(n_iters::Int, n_adapt::Int, delta::Float64, lambda::Float64)\n\nHamiltonian Monte Carlo sampler wiht Dual Averaging algorithm.\n\nUsage:\n\nHMCDA(1000, 200, 0.65, 0.3)\n\nExample:\n\n# Define a simple Normal model with unknown mean and variance.\n@model gdemo(x) = begin\n  s ~ InverseGamma(2,3)\n  m ~ Normal(0,sqrt.(s))\n  x[1] ~ Normal(m, sqrt.(s))\n  x[2] ~ Normal(m, sqrt.(s))\n  return s, m\nend\n\nsample(gdemo([1.5, 2]), HMCDA(1000, 200, 0.65, 0.3))\n\n\n\n\n\n"
},

{
    "location": "api.html#Turing.IPMCMC",
    "page": "API",
    "title": "Turing.IPMCMC",
    "category": "type",
    "text": "IPMCMC(n_particles::Int, n_iters::Int, n_nodes::Int, n_csmc_nodes::Int)\n\nParticle Gibbs sampler.\n\nUsage:\n\nIPMCMC(100, 100, 4, 2)\n\nExample:\n\n# Define a simple Normal model with unknown mean and variance.\n@model gdemo(x) = begin\n  s ~ InverseGamma(2,3)\n  m ~ Normal(0,sqrt(s))\n  x[1] ~ Normal(m, sqrt(s))\n  x[2] ~ Normal(m, sqrt(s))\n  return s, m\nend\n\nsample(gdemo([1.5, 2]), IPMCMC(100, 100, 4, 2))\n\n\n\n\n\n"
},

{
    "location": "api.html#Turing.IS",
    "page": "API",
    "title": "Turing.IS",
    "category": "type",
    "text": "IS(n_particles::Int)\n\nImportance sampling algorithm object.\n\nn_particles is the number of particles to use\n\nUsage:\n\nIS(1000)\n\nExample:\n\n# Define a simple Normal model with unknown mean and variance.\n@model gdemo(x) = begin\n  s ~ InverseGamma(2,3)\n  m ~ Normal(0,sqrt.(s))\n  x[1] ~ Normal(m, sqrt.(s))\n  x[2] ~ Normal(m, sqrt.(s))\n  return s, m\nend\n\nsample(gdemo([1.5, 2]), IS(1000))\n\n\n\n\n\n"
},

{
    "location": "api.html#Turing.MH",
    "page": "API",
    "title": "Turing.MH",
    "category": "type",
    "text": "MH(n_iters::Int)\n\nMetropolis-Hasting sampler.\n\nUsage:\n\nMH(100, (:m, (x) -> Normal(x, 0.1)))\n\nExample:\n\n# Define a simple Normal model with unknown mean and variance.\n@model gdemo(x) = begin\n  s ~ InverseGamma(2,3)\n  m ~ Normal(0,sqrt(s))\n  x[1] ~ Normal(m, sqrt(s))\n  x[2] ~ Normal(m, sqrt(s))\n  return s, m\nend\n\nsample(gdemo([1.5, 2]), MH(1000, (:m, (x) -> Normal(x, 0.1)), :s)))\n\n\n\n\n\n"
},

{
    "location": "api.html#Turing.NUTS",
    "page": "API",
    "title": "Turing.NUTS",
    "category": "type",
    "text": "NUTS(n_iters::Int, n_adapt::Int, delta::Float64)\n\nNo-U-Turn Sampler (NUTS) sampler.\n\nUsage:\n\nNUTS(1000, 200, 0.6j_max)\n\nExample:\n\n# Define a simple Normal model with unknown mean and variance.\n@model gdemo(x) = begin\n  s ~ InverseGamma(2,3)\n  m ~ Normal(0,sqrt.(s))\n  x[1] ~ Normal(m, sqrt.(s))\n  x[2] ~ Normal(m, sqrt.(s))\n  return s, m\nend\n\nsample(gdemo([1.j_max, 2]), NUTS(1000, 200, 0.6j_max))\n\n\n\n\n\n"
},

{
    "location": "api.html#Turing.PG",
    "page": "API",
    "title": "Turing.PG",
    "category": "type",
    "text": "PG(n_particles::Int, n_iters::Int)\n\nParticle Gibbs sampler.\n\nUsage:\n\nPG(100, 100)\n\nExample:\n\n# Define a simple Normal model with unknown mean and variance.\n@model gdemo(x) = begin\n  s ~ InverseGamma(2,3)\n  m ~ Normal(0,sqrt.(s))\n  x[1] ~ Normal(m, sqrt.(s))\n  x[2] ~ Normal(m, sqrt.(s))\n  return s, m\nend\n\nsample(gdemo([1.5, 2]), PG(100, 100))\n\n\n\n\n\n"
},

{
    "location": "api.html#Turing.PMMH",
    "page": "API",
    "title": "Turing.PMMH",
    "category": "type",
    "text": "PMMH(n_iters::Int, smc_alg:::SMC, parameters_algs::Tuple{MH})\n\nParticle independant Metropolis–Hastings and Particle marginal Metropolis–Hastings samplers.\n\nUsage:\n\nalg = PMMH(100, SMC(20, :v1), MH(1,:v2))\nalg = PMMH(100, SMC(20, :v1), MH(1,(:v2, (x) -> Normal(x, 1))))\n\n\n\n\n\n"
},

{
    "location": "api.html#Turing.SGHMC",
    "page": "API",
    "title": "Turing.SGHMC",
    "category": "type",
    "text": "SGHMC(n_iters::Int, learning_rate::Float64, momentum_decay::Float64)\n\nStochastic Gradient Hamiltonian Monte Carlo sampler.\n\nUsage:\n\nSGHMC(1000, 0.01, 0.1)\n\nExample:\n\n@model example begin\n  ...\nend\n\nsample(example, SGHMC(1000, 0.01, 0.1))\n\n\n\n\n\n"
},

{
    "location": "api.html#Turing.SGLD",
    "page": "API",
    "title": "Turing.SGLD",
    "category": "type",
    "text": "SGLD(n_iters::Int, step_size::Float64)\n\nStochastic Gradient Langevin Dynamics sampler.\n\nUsage:\n\nSGLD(1000, 0.5)\n\nExample:\n\n@model example begin\n  ...\nend\n\nsample(example, SGLD(1000, 0.5))\n\n\n\n\n\n"
},

{
    "location": "api.html#Turing.SMC",
    "page": "API",
    "title": "Turing.SMC",
    "category": "type",
    "text": "SMC(n_particles::Int)\n\nSequential Monte Carlo sampler.\n\nUsage:\n\nSMC(1000)\n\nExample:\n\n# Define a simple Normal model with unknown mean and variance.\n@model gdemo(x) = begin\n  s ~ InverseGamma(2,3)\n  m ~ Normal(0,sqrt.(s))\n  x[1] ~ Normal(m, sqrt.(s))\n  x[2] ~ Normal(m, sqrt.(s))\n  return s, m\nend\n\nsample(gdemo([1.5, 2]), SMC(1000))\n\n\n\n\n\n"
},

{
    "location": "api.html#Samplers-1",
    "page": "API",
    "title": "Samplers",
    "category": "section",
    "text": "Sampler\nGibbs\nHMC\nHMCDA\nIPMCMC\nIS\nMH\nNUTS\nPG\nPMMH\nSGHMC\nSGLD\nSMC"
},

{
    "location": "api.html#Index-1",
    "page": "API",
    "title": "Index",
    "category": "section",
    "text": ""
},

]}
