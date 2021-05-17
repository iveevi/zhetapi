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
    os.system('mkdir -p debug/')
    os.system('[ -f libzhp.a ] && mv libzhp.a debug/')
    os.system('[ -f libzhp.so ] && mv libzhp.so debug/')


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

def list_modes(args):
    print('The following are available run modes:')

    for key in modes.keys():
        print('\t' + key)

def install(args):
    print("Installing...")

    os.system('mkdir -p bin')
    os.system('mkdir -p include')

    make_target(args.threads, 'zhetapi zhp-shared zhp-static')

    os.system('mv zhetapi bin/')
    os.system('mv libzhp.* bin/')

    ret = os.system('ln -s -f $PWD/bin/zhetapi /usr/local/bin/zhetapi')
    if ret != 0:
        clean_and_exit(-1)
    
    ret = os.system('ln -s -f $PWD/engine /usr/local/include/zhetapi')
    if ret != 0:
        clean_and_exit(-1)

    ret = os.system('ln -s -f $PWD/bin/libzhp.so /usr/local/lib/libzhp.so')
    if ret != 0:
        clean_and_exit(-1)

    ret = os.system('ln -s -f $PWD/bin/libzhp.a /usr/local/lib/libzhp.a')
    if ret != 0:
        clean_and_exit(-1)

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

    ret = os.system('cp -r $PWD/include /usr/local/include/zhp')
    if ret != 0:
        clean_and_exit(-1)


def zhetapi_normal(args):
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

def zhetapi_profile(args):
    make_target(args.threads, 'zhetapi', args.mode)

    # Use the benchmark test for profiling
    file = 'testing/benchmarks/base_bench.zhp'

    ret = os.system('{exe}zhetapi {file} -L include'.format(
        exe=modes[args.mode],
        file=file
    ))
    
    if (ret != 0):
        clean_and_exit(-1)
    
    ret = os.system('{cmd}'.format(cmd=post[args.mode]))
    
    if (ret != 0):
        clean_and_exit(-1)

    os.system('mkdir -p debug/')
    os.system('mv zhetapi debug/')

    if (ret != 0):
        clean_and_exit(-1)

def zhetapi(args):
    if args.mode == 'profile':
        zhetapi_profile(args)
    else:
        zhetapi_normal(args)

def clang(args):
    ret = os.system('clang-tidy-8 engine/* source/* -- -I engine -I glad')

    if (ret != 0):
        clean_and_exit(-1)

def base_bench(args):
    # Run bench
    start_t = time.time()

    os.system('./bin/zhetapi testing/benchmarks/base_bench.zhp')

    end_t = time.time()

    print(f'\nEXECUTION TIME: {1000 * (end_t - start_t)} ms')

def python_bench(args):
    # Run bench for zhetapi-lang
    print('ZHETAPI-LANG:')

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

    print(f'\nZHETAPI-LANG EXECUTION TIME:\t{1000 * zhp_t:.2f} ms')
    print(f'PYTHON EXECUTION TIME:\t\t{1000 * py_t:.2f} ms')
    print(f'RATIO (ZHP/PY):\t\t\t{100 * zhp_t/py_t:.2f}%')

special = {
    'install': install,
    'zhetapi': zhetapi,
    'clang' : clang,
    'list': list_features,
    'modes' : list_modes,
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
