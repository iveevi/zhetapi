#!/bin/bash

cmake .

mkdir -p bin

make zhp

make port

mv port bin/

if [ -f "libzhp.so" ] || [ -f "libzhp.a" ]; then
	mv libzhp.* bin/
fi

./bin/port
