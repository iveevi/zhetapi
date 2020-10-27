#!/bin/sh
echo "PATH=${PATH}:$(PWD)/build" >> .env

source .env

make install
