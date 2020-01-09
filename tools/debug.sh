# !/bin/bash

if [ "$#" = 1 ]; then
	"$1" -d
else
	./runner -d
fi