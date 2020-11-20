#!/bin/bash

cmake -DCMAKE_BUILD_TYPE=RELEASE .

mkdir -p bin

make zhp

make port

mv port bin/