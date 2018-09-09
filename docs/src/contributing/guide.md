# Contributing

Turing is an open source project. If you feel you have some relevant skills and are interested in contributing, then please do get in touch. You can contribute by opening issues on Github or implementing things yourself and making a pull request. We would also appreciate example models written using Turing.

Turing has a [style guide](../style_guide.html). It is not strictly necessary to review it, but particularly adventurous contributors may enjoy reviewing it.

## How to Contribute

### Getting started
* [Fork this repository](https://github.com/TuringLang/Turing.jl#fork-destination-box)
* Clone your fork on your local machine: `git clone https://github.com/your_username/Turing.jl`
* Add a remote corresponding to this repository:
`git remote add upstream https://github.com/TuringLang/Turing.jl`


### What can I do ?
Look at the [issues](https://github.com/TuringLang/Turing.jl/issues) page to find an outstanding issue. For instance, you could implement new features, fix bugs or write example models.

### Git workflow
* Make sure that your local master branch is up to date with this repository's one ([for more details](https://stackoverflow.com/questions/7244321/how-do-i-update-a-github-forked-repository)):
```bash
git fetch upstream
git checkout master
git rebase upstream/master
```
* Create a new branch: `git checkout -b branch_name` (usually use `feature-issue_id` or `bugfix-issue_id`)
* Do your stuff: `git add ...`, `git commit -m '...'`
* Push your local branch to your fork of this repository: `git push --set-upstream origin branch_name`

### Make a pull request
* Create a pull request by going to [this repository front page](https://github.com/TuringLang/Turing.jl) and selecting `Compare & pull request`
* If related to a specific issue, link the pull request link in that issue, and in the pull request also link the issue
