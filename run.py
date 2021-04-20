#!/usr/bin/python3

import argparse
import os

# Build the parser
parser = argparse.ArgumentParser()

# {TARGET} -m {EXECUTOR} -j{THREADS}
parser.add_argument("target", help="Database name")
parser.add_argument("-m", "--mode", help="Execution mode", default='')
parser.add_argument("-j", "--threads", help="Number of concurrent threads", type=int, default=8)

# Cleaning
def clean():
    os.system('[ -f libzhp.* ] && mv libzhp.* debug/')

def clean_and_exit(sig):
    clean()

    exit(sig)

# Compilation
def make_target(threads, target, mode=''):
	if mode in ['gdb', 'valgrind', 'profile']:
		ret = os.system('cmake -DCMAKE_BUILD_TYPE=Debug .')
	elif mode in ['warn']:
		ret = os.system('cmake -DCMAKE_BUILD_TYPE=Warn .')
	else:
		ret = os.system('cmake -DCMAKE_BUILD_TYPE=Release .')

	if ret != 0:
		clean_and_exit(-1)

	ret = os.system('make -j{threads} {target}'.format(
		threads=threads,
		target=target
	))

	if ret != 0:
		clean_and_exit(-1)

# Execution modes
modes = {
	'': './',
	'gdb': 'gdb ',
	'warn': './',
	'valgrind': 'valgrind --leak-check=full --track-origins=yes ./',
        'profile': 'valgrind --tool=callgrind --callgrind-out-file=callgrind.out ./'
}

# Post scripts
post = {
        'profile': 'kcachegrind callgrind.out && rm callgrind.out'
}

# Feature testing
features = {
        'imv': 'image view test'
}

# Special targets
def list(args):
        print('The following are avaiable feature tests to run:')

        for key in features.keys():
            print('\t' + key + ': ' + features[key])

def install(args):
	print("Installing...")

	os.system('mkdir -p bin')
	os.system('mkdir -p include')

	make_target(args.threads, 'zhetapi zhp-shared zhp-static')

	os.system('mv zhetapi bin/')
	os.system('mv libzhp.* bin/')

	print(50 * '=' + "\nCompiling libraries...\n" + 50 * '=')
	ret1 = os.system('./bin/zhetapi -v -c	\
		lib/io/io.cpp		\
		lib/io/formatted.cpp	\
		lib/io/file.cpp		\
		-o include/io.zhplib')

	ret2 = os.system('./bin/zhetapi -v -c	\
		lib/math/math.cpp	\
		-o include/math.zhplib')

	print("\n" + 50 * '=' + "\nDisplaying symbols\n" + 50 * '=')
	os.system('./bin/zhetapi -d include/io.zhplib')
	os.system('./bin/zhetapi -d include/math.zhplib')

	if (ret1 != 0) or (ret2 != 0):
		clean_and_exit(-1)

def zhetapi(args):
    make_target(args.threads, 'zhetapi', args.mode)

    file = 'samples/zhp/simple.zhp'

    ret = 0
    if args.mode == '':
        ret = os.system('{exe}zhetapi {file} -L include'.format(
            exe=modes[args.mode],
            file=file
        ))
    else:
        ret = os.system('{exe}zhetapi'.format(
            exe=modes[args.mode]
        ))

    os.system('mkdir -p debug/')
    os.system('mv zhetapi debug/')

    if (ret != 0):
        clean_and_exit(-1)

special = {
	'install': install,
	'zhetapi': zhetapi,
        'list': list
}

# Preprocessing
targets = [key for key in features.keys()]

for filename in os.listdir("cmake"):
    if filename.endswith(".cmake"):
        targets.append(filename[:-6])

args = parser.parse_args()

# Execute
if args.target in special.keys():
    special[args.target](args)
elif args.target in targets:
    make_target(args.threads, args.target, args.mode)

    ret1 = os.system('{exe}{target}'.format(
            exe=modes[args.mode],
            target=args.target
    ))

    # Check for any post processing scripts
    ret2 = 0
    if args.mode in post:
        ret2 = os.system('{cmd}'.format(cmd=post[args.mode]))

    os.system('mkdir -p debug/')

    os.system('mv {target} debug/'.format(
            target=args.target
    ))

    if (ret1 != 0) or (ret2 != 0):
        clean_and_exit(-1)

clean()