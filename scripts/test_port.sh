#!/bin/bash

if [ "$1" = "gdb" ] || [ "$1" = "valgrind" ]; then
	cmake -DCMAKE_BUILD_TYPE=DEBUG .
else
	cmake -DCMAKE_BUILD_TYPE=RELEASE .
fi

mkdir -p bin

make zhp

make port

mv port bin/

if [ "$1" = "gdb" ]; then
	gdb ./bin/port
elif [ "$1" = "valgrind" ]; then
	valgrind ./bin/port
else
	./bin/port
fi

if [ -f "libzhp.so" ] || [ -f "libzhp.a" ]; then
	mv libzhp.* bin/
fi