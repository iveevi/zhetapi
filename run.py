#!/bin/python3

import argparse
import sys
import os

# Build the parser
parser = argparse.ArgumentParser()

# {TARGET} -m {EXECUTOR} -j{THREADS}
parser.add_argument("target", help="Database name")
parser.add_argument("-m", "--mode", help="Execution mode", default='')
parser.add_argument("-j", "--threads", help="Number of concurrent threads", type=int, default=8)

# Compilation
def compile(threads, target):
	os.system('cmake .')

	os.system('make -j{threads} {target}'.format(
		threads=threads,
		target=target
	))

# Execution modes
modes = {
	'': './',
	'gdb': 'gdb ',
	'valgrind': 'valgrind --leakcheck=full '
}

# Special targets
def install(args):
	print("Installing...")

	compile(args.threads, 'czhp')
	compile(args.threads, 'zhp-shared')
	compile(args.threads, 'zhp-static')

	os.system('mkdir -p bin')
	os.system('mv czhp bin/')
	os.system('mv libzhp.* bin/')

	os.system('mkdir -p include')

	print(50 * '=' + "\nCompiling libraries...\n" + 50 * '=')
	os.system('./bin/czhp -v -c	\
		lib/io/io.cpp		\
		lib/io/formatted.cpp	\
		lib/io/file.cpp		\
		-o include/io.zhplib')
	
	os.system('./bin/czhp -v -c	\
		lib/math/math.cpp	\
		-o include/math.zhplib')
	
	print("\n" + 50 * '=' + "\nDisplaying symbols\n" + 50 * '=')
	os.system('./bin/czhp -d include/io.zhplib')
	os.system('./bin/czhp -d include/math.zhplib')

def czhp(args):
	compile(args.threads, 'czhp')

	file = 'samples/zhp/simple.zhp'

	os.system('{exe}czhp {file} -L include'.format(
		exe=modes[args.mode],
		file=file
	))

	os.system('mkdir -p debug/')
	os.system('mv czhp debug/')

special = {
	'install': install,
	'czhp': czhp
}

# Preprocessing
targets = []

for filename in os.listdir("cmake"):
	if filename.endswith(".cmake"):
		targets.append(filename[:-6])

args = parser.parse_args()

# Execute
if args.target in special.keys():
	special[args.target](args)
elif args.target in targets:
	compile(args.threads, args.target)

	os.system('{exe}{target}'.format(
		exe=modes[args.mode],
		target=args.target
	))

	os.system('mkdir -p debug/')
	
	os.system('mv {target} debug/'.format(
		target=args.target
	))

os.system('[ -f libzhp.* ] && mv libzhp.* debug/')