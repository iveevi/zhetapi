#!/bin/bash

cmake .

mkdir -p bin

make zhp

make zhetapi

mv zhetapi bin/

if [ -f "libzhp.so" ] || [ -f "libzhp.a" ]; then
	mv libzhp.* bin/
fi