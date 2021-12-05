#!/bin/bash
black "."
isort .
#pdoc --force --html -o docs src
#mv docs/FashionMNIST_G25/index.html docs/index.md
#mv docs/sFashionMNIST_G2/* docs/
jupytext --to notebook src/*.py
if [[ ! -z $1 ]]; then
        git add . && git commit -m $1 && git push
fi
