import subprocess
import os
import signal
import time
import struct

from subprocess import *
from flask import Flask
from flask import request
from flask import render_template

BUILD_DIR = "/home/ram/zhetapi/build/"
PIPE_DIR = "/home/ram/null/"

print("started....");
fout = open(PIPE_DIR + "web2drv", "wb")

print("[App]: Opened pipes...")

def compute(input):
    print("[App]: In computer")
    print("[App]: Packaging " + input)

    in_bytes = struct.pack("<I", len(input))
    in_bytes += bytes(input, 'utf-8')

    print("[App]: Sending " + str(bytearray(in_bytes)))
    print("length is " + str(len(input)))

    fout.write(bytearray(in_bytes))
    fout.flush()

    print("opening fin");
    fin = open(PIPE_DIR + "drv2web", "r")
    # print("fin opened")
    # x = struct.unpack("<I", fin.read())[0]
    # print("drv acked " + str(x) + " bytes")
    x = fin.readline()
    print("got: " + x)
    fin.close()

    return x

compute("23 / 54")
compute("e^2")
# compute("")

# fout.close()

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

driver.wait()
