#!/usr/bin/python3

import argparse
import os
import time

# Build the parser
parser = argparse.ArgumentParser()

# {TARGET} -m {EXECUTOR} -j{THREADS}
parser.add_argument("target", help="Database name")
parser.add_argument("-m", "--mode", help="Execution mode", default='')
parser.add_argument("-j", "--threads",
                    help="Number of concurrent threads", type=int, default=8)

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


def list_features(args):
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
    ret = os.system('./bin/zhetapi -v -c	\
		lib/io/io.cpp		\
		lib/io/formatted.cpp	\
		lib/io/file.cpp		\
		-o include/io.zhplib')
    if ret != 0:
        clean_and_exit(-1)

    ret = os.system('./bin/zhetapi -v -c	\
		lib/math/math.cpp	\
		-o include/math.zhplib')
    if ret != 0:
        clean_and_exit(-1)

    print("\n" + 50 * '=' + "\nDisplaying symbols\n" + 50 * '=')

    ret = os.system('./bin/zhetapi -d include/io.zhplib')
    if ret != 0:
        clean_and_exit(-1)

    ret = os.system('./bin/zhetapi -d include/math.zhplib')
    if ret != 0:
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

def clang(args):
    ret = os.system('clang-tidy-8 engine/* source/* -- -I engine -I glad')

    if (ret != 0):
        clean_and_exit(-1)

def base_bench(args):
    print('Installing interpreter...')

    # Make directories and compile interpreter
    os.system('mkdir -p bin')

    make_target(args.threads, 'zhetapi zhp-shared zhp-static')

    os.system('mv zhetapi bin/')
    os.system('mv libzhp.* bin/')

    # Run bench
    start_t = time.time()

    os.system('./bin/zhetapi testing/benchmarks/base_bench.zhp')

    end_t = time.time()

    print(f'\nEXECUTION TIME: {1000 * (end_t - start_t)} ms')

def python_bench(args):
    print('Installing interpreter...')

    # Make directories and compile interpreter
    os.system('mkdir -p bin')

    make_target(args.threads, 'zhetapi zhp-shared zhp-static')

    os.system('mv zhetapi bin/')
    os.system('mv libzhp.* bin/')

    # Run bench for zhetapi-lang
    print('ZHEPATI-LANG:')

    start_t = time.time()

    os.system('./bin/zhetapi testing/benchmarks/relative_bench.zhp')

    end_t = time.time()

    zhp_t = end_t - start_t
    
    # Run bench for python
    print('\nPYTHON:')

    start_t = time.time()

    os.system('python3 testing/benchmarks/relative_bench.py')

    end_t = time.time()

    py_t = end_t - start_t

    print(f'\nZHETAPI-LANG EXECUTION TIME: {1000 * zhp_t} ms')
    print(f'PYTHON EXECUTION TIME: {1000 * py_t} ms')

special = {
    'install': install,
    'zhetapi': zhetapi,
    'clang' : clang,
    'list': list_features,
    'base_bench' : base_bench,
    'python_bench' : python_bench
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
else:
    print(f'Unknown target \"{args.target}\"')

clean()
