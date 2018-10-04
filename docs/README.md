# Building the Documentation Site Locally

The documentation site is built by Travis and served from the `gh-pages` branch of this repo. If you wish to edit the website 
locally and review your changes, you will need to do the following:

1. Install Jekyll by following [the relevant guide](https://jekyllrb.com/docs/installation/) for your operating system.

2. Navigate in a terminal to `Turing.jl/docs/site`.

3. Type `jekyll serve` if you wish to review the changes in a browser. Typically the website will be served on `localhost:4000`,
which you can visit in your browser.

4. Alternatively, if you simply want to build the site files into a static site, you can use `jekyll build`. 

The files in `docs/site` may be out of date with the website. If this is the case, execute the `make.jl` script found 
at `Turing.jl/docs/make.jl`, which will run all the documents in the `docs/src` folder through a markdown processor and place 
them into the `docs/site/` folder. Any files located in the `docs/site/_docs/` or `docs/site/_tutorials/` directories 
should **not** be edited directly, as they will be overwritten by the versions in the `docs/src/` directory. Edits must
be made to the `docs/src/` versions.
