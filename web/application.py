import subprocess

from flask import Flask
from flask import request
from flask import render_template

app = Flask(__name__)

@app.route('/')
def form():
    subprocess.run(["./run"])
    return render_template('form.html')

@app.route('/', methods=['POST'])
def post():
    text = request.form['text']

    sin = open("../build/driver.in", "w")
    sin.write(text)
    sin.close()

    sin = open("../build/texifier.in", "w")
    sin.write(text)
    sin.close()

    subprocess.run(["./../build/driver"])
    subprocess.run(["./../build/texifier"])

    sout = open("../build/driver.out", "r")
    out = sout.read()
    sout.close()

    sout = open("../build/texifier.out", "r")
    text = sout.read()
    sout.close()

    return render_template('form.html', input = text, out = out)
