#!/usr/bin/python3

import argparse
import os

# Build the parser
parser = argparse.ArgumentParser()

# {TARGET} -m {EXECUTOR} -j{THREADS}
parser.add_argument("target", help="Database name")
parser.add_argument("-m", "--mode", help="Execution mode", default='')
parser.add_argument("-j", "--threads", help="Number of concurrent threads", type=int, default=8)

# Compilation
def make_target(threads, target, mode=''):
	if mode in ['gdb', 'valgrind']:
		ret = os.system('cmake -DCMAKE_BUILD_TYPE=Debug .')
	else:
		ret = os.system('cmake -DCMAKE_BUILD_TYPE=Release .')

	if ret != 0:
		exit(-1)

	ret = os.system('make -j{threads} {target}'.format(
		threads=threads,
		target=target
	))

	if ret != 0:
		exit(-1)

# Execution modes
modes = {
	'': './',
	'gdb': 'gdb ',
	'valgrind': 'valgrind --leak-check=full --track-origins=yes '
}

# Special targets
def install(args):
	print("Installing...")

	make_target(args.threads, 'czhp zhp-shared zhp-static')
	# make_target(args.threads, 'zhp-shared')
	# make_target(args.threads, 'zhp-static')

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
	make_target(args.threads, 'czhp', args.mode)

	file = 'samples/zhp/simple.zhp'

	if args.mode == '':
		os.system('{exe}czhp {file} -L include'.format(
			exe=modes[args.mode],
			file=file
		))
	else:
		os.system('{exe}czhp'.format(
			exe=modes[args.mode]
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
	make_target(args.threads, args.target, args.mode)

	os.system('{exe}{target}'.format(
		exe=modes[args.mode],
		target=args.target
	))

	os.system('mkdir -p debug/')

	os.system('mv {target} debug/'.format(
		target=args.target
	))

os.system('[ -f libzhp.a ] && mv libzhp.a debug/')
os.system('[ -f libzhp.os ] && mv libzhp.os debug/')
