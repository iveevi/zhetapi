#!/bin/bash

set -e

if [ "$1" = "-debug" ] || [ "$1" = "-d" ]; then
	g++ ../src/main.cpp -DDEBUG=0 -o ../exe/zhetapi
	../exe/zhetapi
else
	g++ ../src/main.cpp -o ../exe/zhetapi
	../exe/zhetapi
fi
