#!/bin/bash
black "."
isort .
python3 -m pdoc --force --html --output-dir docs src/
mv docs/src/index.html docs/index.md
mv docs/src/* docs/
jupytext --to notebook src/*.py
if [[ ! -z $1 ]]; then
        git add . && git commit -m $1 && git push
fi
