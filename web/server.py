import os
import time
import struct

from subprocess import *

proc = Popen("./a.out")

while not os.path.exists("output"):
    pass

while not os.path.exists("input"):
    pass

fin = open("output", "r")
fout = open("input", "wb")

user = "?"

while True:
    user = input("$ ")

    if len(user) == 0:
        break

    str_len = len(user)
    str_bytes = struct.pack("<I", str_len)
    str_bytes += bytes(user, 'utf-8')

    out = bytearray(str_bytes)

    print("\tSending: " + str(out))

    fout.write(out)
    fout.flush()

    user = fin.readline()

    print("\tReceived: " + user)

fout.write(bytearray(0))
fout.close()
fin.close()

proc.wait()
