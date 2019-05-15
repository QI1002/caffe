#!/usr/bin/env sh
set -e

./build/tools/caffe-d train --solver=examples/mnist/lenet_solver.prototxt $@
