#!/bin/python3

import sys

args = sys.argv

print(args)

modes = {'gdb': '', 'valgrind': '--leakcheck=full'}

print(modes)