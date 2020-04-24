import struct
import os

fin = os.open("driver.out", os.O_RDONLY)
fout = os.open("driver.in", os.O_WRONLY)

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
