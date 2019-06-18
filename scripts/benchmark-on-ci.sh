#!/bin/bash
set -v

BASE_BRANCH=master

if [[ $TRAVIS == true ]]; then
    if [[ $TRAVIS_PULL_REQUEST == true ]]; then
        CURRENT_BRANCH=$TRAVIS_PULL_REQUEST_BRANCH
    else
        CURRENT_BRANCH=$TRAVIS_BRANCH
    fi
else
    CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
fi
        
COMMIT_MSG=$(git show -s --format="%s")

if [[ $COMMIT_MSG != *"[bm]"* ]]; then
    echo "skipping the benchmark jobs."
    # exit 0
fi

git config remote.origin.fetch '+refs/heads/*:refs/remotes/origin/*'
git fetch --all --unshallow

git clone https://github.com/TuringLang/TuringBenchmarks.git ../TuringBenchmarks
git -C ../TuringBenchmarks checkout -b travis-ci origin/travis-ci

unset JULIA_PROJECT
cat > pre.jl <<EOF
using Pkg
# Pkg.instantiate()
try pkg"develop ." catch end
try pkg"develop ." catch end
try pkg"build Turing" catch end
using Turing
try pkg"develop ../TuringBenchmarks" catch end
try pkg"develop ../TuringBenchmarks" catch end
pkg"add SpecialFunctions"
using TuringBenchmarks
EOF

julia pre.jl

cat > run.jl <<EOF
using TuringBenchmarks.Runner

report_path = Runner.local_benchmark("Travis-CI", ("master", "$CURRENT_BRANCH"))

@show report_path
EOF

cat run.jl
julia run.jl
