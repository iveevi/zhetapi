#!/usr/bin/python3
import os
import sys

# TODO: integrate into a parsing library functionality
file = sys.argv[1]
ext = file[-5:]

if ext != '.nota':
	print('Invalid file, expected *.nota')
	sys.exit(-1)

counter = 0
with open(file) as fin:
	lines = fin.readlines()

lines = [line.strip() for line in lines]
lines = [line for line in lines if len(line) > 0 and line[0] != '#']

whole = ''.join(lines)
whole = [token.strip() for token in whole.split(',')]

print(whole)

# TODO: allow for custom class (to replace LexClass)
lines = []
for token in whole:
	struct = 'struct ' + token + ' : public LexClass <' + str(counter) + '> {};\n'
	counter += 1
	lines.append(struct)

fdir, fname = os.path.split(file)
with open(fdir + '/' + fname[:-5] + '_nota.hpp', 'w') as fout:
	fout.writelines(lines)