#!/bin/sh
echo "PATH=${PATH}:$(PWD)/apps" >> .env

source .env

cmake .
make

mkdir -p apps
mv zhetapi apps/
