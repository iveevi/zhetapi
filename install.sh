#!/bin/sh

# Organize headers
if [ -d zhetapi-headers ]; then
	rm -rf zhetapi-headers;
fi

mkdir zhetapi-headers

cp engine/* zhetapi-headers
cp -r inc/* zhetapi-headers

# Register apps directory
echo "PATH=${PATH}:$(pwd)/apps" >> .env

source .env

# Compile and move apps
cmake .
make

mkdir -p apps
mv zhetapi apps/