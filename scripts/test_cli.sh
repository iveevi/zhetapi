#!/bin/bash
set -e

cmake .

mkdir -p bin

make -j8 zhp

make -j8 zhetapi

mv zhetapi bin/

if [ -f "libzhp.so" ] || [ -f "libzhp.a" ]; then
	mv libzhp.* bin/
fi
