import subprocess
import os
import signal
import time
import struct

from subprocess import *

counter = 1

print("[App]: Reseting shared files...")
os.system("rm -rf ../build")

print("[App]: Building executables...")
os.system("./run")

print("[App]: Launching driver...")
driver = subprocess.Popen("./../build/driver")
# driver = Popen("./a.out")

while not os.path.exists("../build/driver.in"):
    pass

while not os.path.exists("../build/driver.out"):
    pass

fin = open("../build/driver.in", "r")
fout = open("../build/driver.out", "wb")

"""while not os.path.exists("input"):
    pass

while not os.path.exists("output"):
    pass

fin = open("output", "r")
fout = open("input", "wb")"""

print("[App]: Opened pipes...")

# Regular text to LaTeX converter
def convert(input):
    print("[App]: In converter")
    sin = open("../build/texifier.in", "w")
    sin.write(input)
    sin.close()

    subprocess.run(["./../build/texifier"])

    sout = open("../build/texifier.out", "r")
    text = sout.read()
    sout.close()

    return text

# Helper method to call driver
def compute(input):
    global counter
    
    print("[App]: In computer")

    # print("[App]: Waiting to open pipes...\n")

    print("[App]: Packaging " + input)

    in_bytes = struct.pack("<I", len(input))
    in_bytes += bytes(input, 'utf-8')

    print("[App]: Sending " + str(bytearray(in_bytes)))

    fout.write(bytearray(in_bytes))
    fout.flush()

    out = fin.readline()

    print("[App]: Receiving " + out)

    counter += 1
    
    return out

from flask import Flask
from flask import request
from flask import render_template

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/calculate')
def show_calculate_view():
    return render_template('calculate.html')

@app.route('/calculate', methods=['POST'])
def calculate_post():
    text = request.form['text']
    out = compute(text)
    text = convert(text)
    return render_template('calculate.html', input = text, out = out)

logs = []

@app.route('/session')
def show_session_view():
    return render_template('session.html')

@app.route('/session', methods=['POST'])
def session_post():
    text = request.form['text']
    out = compute(text)
    text = convert(text)
    
    logs.insert(0, text + " = \\textbf{" + out + "}");
    return render_template('session.html', logs = logs)

print("[App]: After the program...\n")
# fout.write(bytearray(0))
# fout.close()
# fin.close()

# driver.wait()
