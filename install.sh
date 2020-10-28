#!/bin/sh
echo "PATH=${PATH}:$(pwd)/apps" >> .env

source .env

cmake .
make

mkdir -p apps
mv zhetapi apps/
