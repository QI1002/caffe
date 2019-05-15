#!/usr/bin/env sh
set -e

TOOLS=./build/tools

$TOOLS/caffe-d train --solver=examples/siamese/mnist_siamese_solver.prototxt $@
