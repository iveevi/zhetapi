import subprocess
import os
import signal
import time

import sample

counter = 1

print("[App]: Reseting shared files...")
os.system("rm -rf ../build")

print("[App]: Building executables...")
os.system("./run")

print("[App]: Launching driver...")
# driver = subprocess.Popen("./../build/driver", shell = False)

sin = open("../build/pid", "w")
sin.write(str(os.getpid()))
sin.close()

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

done = False
def switch(a, b):
    print("[App]: Received signal...")
    done = True

# Helper method to call driver
def compute(input):
    global counter

    sin = open("../build/driver.in", "a")
    sin.write("#" + str(counter) + "\t" + input + "\n")
    sin.close()

    counter += 1

    done = False

    # driver.send_signal(signal.SIGHUP)
    """ print("[App]: Sent signal, waiting for response...")

    signal.signal(signal.SIGINT, switch)
    while not done:
        pass """

    # sample.get()

    # print("[App]: Got signal, proceeding...")

    sout = open("../build/driver.out", "r")
    out = sout.read()
    sout.close()
    
    return out

from flask import Flask
from flask import request
from flask import render_template

app = Flask(__name__)
signal.signal(signal.SIGINT, switch)

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

# driver.send_signal(signal.SIGKILL)
