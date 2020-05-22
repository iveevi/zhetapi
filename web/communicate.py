import struct
import subprocess

PIPE_DIR = "/home/ram/zhetapi/build/"

fout = open(PIPE_DIR + "driver_in", "wb")

# Regular text to Zhetapi Engine
def compute(input):
    in_bytes = struct.pack("<I", len(input))
    in_bytes += bytes(input, 'utf-8')

    fout.write(bytearray(in_bytes))
    fout.flush()

    fin = open(PIPE_DIR + "driver_out", "r")

    x = fin.readline()
    fin.close()

    return x

# Regular text to LaTeX converter
def convert(input):
    sin = open("../build/texifier.in", "w")
    sin.write(input)
    sin.close()

    subprocess.run(["./../build/texifier"])

    sout = open("../build/texifier.out", "r")
    text = sout.read()
    sout.close()

    return text
