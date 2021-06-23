#!/usr/bin/python3

# TODO: add cleanup for sudo install (CMakeFiles/ bin/ debug/)
# TODO: Install into different directory
import threading
import argparse
import os
import time

# Execution modes
modes = {
    '': './',
    'gdb': 'gdb ',
    'cuda-gdb': 'cuda-gdb ',
    'warn': './',
    'codecov': './',
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

# Preprocessing
targets = [key for key in features.keys()]
header_tests = []

for filename in os.listdir("cmake"):
    if filename.endswith(".cmake"):
        targets.append(filename[:-6])

for filename in os.listdir("testing/headers"):
    if filename.endswith("_headers.cpp"):
        header_tests.append('testing/headers/' + filename)

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

# Executing shell
def run_and_check(suppress, *args):
    csup = ''
    if suppress:
        csup = ' >> install.log 2>&1'

    for cmd in args:
        ret = os.system(cmd + csup)

        if ret != 0:
            print(f'Error executing command \'{cmd}\'')
            clean_and_exit(-1)

# Run process with a spinner
def spinner_process(function, arguments, msg1, msg2):
    # Spinner string
    spinner = '|\\-/'

    thread = threading.Thread(target=function, args=arguments)
    thread.start();

    index = 0
    while thread.is_alive():
        print(f'{spinner[index]} {msg1}', end='\r')
        index = (index + 1) % len(spinner)

        time.sleep(0.1)

    print(f'{msg2}' + ' ' * 50)


# Compilation
def make_target(threads, target, mode='', suppress=False):
    csup = ''
    if suppress:
        csup = ' >> install.log 2>&1'

    if mode in ['gdb', 'cuda-gdb', 'valgrind', 'profile']:
        ret = os.system(f'cmake -DCMAKE_BUILD_TYPE=Debug . {csup}')
    elif mode in ['warn']:
        ret = os.system(f'cmake -DCMAKE_BUILD_TYPE=Warn . {csup}')
    elif mode in ['codecov']:
        ret = os.system(f'cmake -DCMAKE_BUILD_TYPE=Codecov . {csup}')
    else:
        ret = os.system(f'(cmake -DCMAKE_BUILD_TYPE=Release .) {csup}')

    if ret != 0:
        clean_and_exit(-1)

    ret = os.system(f'make -j{threads} {target} {csup}')

    if ret != 0:
        clean_and_exit(-1)

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
    # TODO: print time stats as well

    # zhetapi
    spinner_process(make_target,
            (args.threads, 'zhetapi', args.mode, True),
            'Compiling ZHP interpreter...',
            'Finished compiling ZHP interpreter.'
    )

    spinner_process(make_target,
            (args.threads, 'zhp-shared', args.mode, True),
            'Compiling libzhp.so...',
            'Finished compiling libzhp.so.'
    )

    spinner_process(make_target,
            (args.threads, 'zhp-static', args.mode, True),
            'Compiling libzhp.a...',
            'Finished compiling libzhp.a.'
    )

    run_and_check(
        False,
        'mkdir -p bin',
	'mkdir -p include',
        'mv zhetapi bin/',
        'mv libzhp.* bin/',
        'ln -s -f $PWD/bin/zhetapi /usr/local/bin/zhetapi',
        'ln -s -f $PWD/engine /usr/local/include/zhetapi',
        'ln -s -f $PWD/bin/libzhp.so /usr/local/lib/libzhp.so',
        'ln -s -f $PWD/bin/libzhp.a /usr/local/lib/libzhp.a',
    )

    print('\nFinished installing essentials, moving on to libraries.\n')

    spinner_process(run_and_check,
            (True, './bin/zhetapi -v -c lib/io/io.cpp lib/io/formatted.cpp lib/io/file.cpp -o include/io.zhplib', ),
            'Compiling IO libary...',
            'Finished compiling IO library.'
    )

    spinner_process(run_and_check,
            (True, './bin/zhetapi -v -c lib/math/math.cpp -o include/math.zhplib', ),
            'Compiling Math libary...',
            'Finished compiling Math library.'
    )

    run_and_check(
            True,
            'cp -Trv include /usr/local/lib/zhp'
    )

    print('\nFinished installing everything. Get start with \'zhetapi -h\'')

def zhetapi_normal(args):
    make_target(args.threads, 'zhetapi', args.mode)

    file = 'samples/zhp/simple.zhp'

    ret = 0
    if args.mode == '':
        ret = os.system('{exe}zhetapi {file}'.format(
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

# TODO: un run and check
def zhetapi_memcheck(args):
    make_target(args.threads, 'zhetapi', args.mode)

    # Use the benchmark test for profiling
    file = 'samples/zhp/simple.zhp'

    ret = os.system('{exe}zhetapi {file}'.format(
        exe=modes[args.mode],
        file=file
    ))

    if (ret != 0):
        clean_and_exit(-1)

    os.system('mkdir -p debug/')
    os.system('mv zhetapi debug/')

    if (ret != 0):
        clean_and_exit(-1)

def zhetapi_profile(args):
    make_target(args.threads, 'zhetapi', args.mode)

    # Use the benchmark test for profiling
    file = 'testing/benchmarks/base_bench.zhp'

    ret = os.system('{exe}zhetapi {file}'.format(
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
    elif args.mode == 'valgrind':
        zhetapi_memcheck(args)
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

def run_header_tests(args):
    run_and_check(False, 'mkdir -p htests')

    for filename in header_tests:
        output = "htests/" + filename[16:-12] + "_htest.out"
        run_and_check(False,
            f'echo \'Running tests for {filename}:\'',
            f'g++-8 {filename} -lzhp -o {output}',
            f'./{output}'
        )

special = {
    'install': install,
    'zhetapi': zhetapi,
    'clang' : clang,
    'list': list_features,
    'modes' : list_modes,
    'base_bench' : base_bench,
    'python_bench' : python_bench,
    'header_tests' : run_header_tests
}

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
