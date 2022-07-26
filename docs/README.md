# Editing the Documentation Site

The Turing site is broken into two different folder structures:

- `docs/src` contains all the _content_ of the site. This is where any
  markdown (`.md`) file is located. When the site is built, the files
  in this directory are run through `Documenter`, as well as some
  Turing-specific pre and postprocessing to get LaTeX and YAML headers
  correct. The folder structure here mimics that of `docs/site`.
- `docs/assets`, `docs/_data`, `docs/_includes`, `docs/_layouts`,
  `docs/pages`, and `docs/_search` contain anything related to the
  _structure_ and _appearance_ of the website. You can find all the
  webpage formatting, site configuration, and navigation configuration
  files here.

Should you need to add a new document to the documentation section or
the tutorials section, this needs to be changed in the file
`docs/_data/toc.yml`.

# Building the Documentation Site Locally

The documentation site is built by Travis and served from the
`gh-pages` branch of this repo. If you wish to edit the website
locally and review your changes, you will need to do the following:

1. Install Jekyll by following [the relevant
   guide](https://jekyllrb.com/docs/installation/) for your operating
   system.

2. Install Julia by following [the relevant
   guide](https://julialang.org/downloads/).

3. Clone the [`turing.ml` repository](https://github.com/TuringLang/turing.ml), 
   where the website is built from.

4. Add your local version of `Turing.jl` to the `turing.ml` environment by running
   `using Pkg; Pkg.dev("path/to/Turing.jl")`.

5. Navigate in a terminal to `turing.ml/`. Run `julia make_local.jl`,
   then you'll get a `_docs` folder full of documents. A `_tutorials`
   folder is also copied from the Turing Tutorial Repo. You might need
   to install `Documenter` and `DocumenterMarkdown` first. You can
   install these packages by running `julia -e 'using
   Pkg;Pkg.add("Documenter");Pkg.add("DocumenterMarkdown");Pkg.add("DynamicHMC");Pkg.add("Turing");Pkg.build()'`.

6. Type `bundle exec jekyll serve` if you wish to review the changes
   in a browser. Typically the website will be served on
   `localhost:4000`, which you can visit in your browser.

7. Alternatively, if you simply want to build the site files into a
   static site, you can use `bundle exec jekyll build`.


The files in `_docs/` may be out of date with the website. If this is
the case, execute the `make_local.jl` script, which will run all the documents in the
`Turing.jl/docs/src` folder through a markdown processor and place them into the
`turing.ml/_docs/` folder. Any files located in the `_docs/` or
`turing.ml/_tutorials/` directories should **not** be edited directly, as
they will be overwritten by the versions in the `Turing.jl/docs/src/`
directory. Edits must be made to the `Turing.jl/docs/src/` versions.

## MacOS Notes
Under MacOS one might need to install the following additional gems's
to have jekyll running as described above.

```
gem install jekyll-paginate
gem install jekyll-sitemap
gem install jekyll-gist
gem install jekyll-feed
gem install jemoji
```

Note: I guess these packages should be installed automatically, but were not in my case.
