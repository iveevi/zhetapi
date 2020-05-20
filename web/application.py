import communicate
import os
import glob

from flask import Flask
from flask import request
from flask import render_template
from flask import send_file
from flask import send_from_directory
from flask import Response

from parse import *

app = Flask(__name__)

app.config["DATA"] = "/home/ram/zhetapi/web/data"

files = glob.glob("/home/ram/zhetapi/web/data/*")
for f in files:
    os.remove(f)


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/graph/<string:date>/<string:ftr>/<string:range>/<string:center>')
def graph(date, ftr, range, center):
    os.system("echo -e \"" + ftr + "\\n" + str(range) + "\\n" + str(center) +
              "\" | /home/ram/zhetapi/build/graph > /home/ram/zhetapi/web/data/graph_" + date + "_" + range + "_" + center)
    return send_from_directory(app.config["DATA"], filename="graph_%s_%s_%s" % (date, str(range), str(center)), as_attachment=True)


@app.route('/calculate')
def show_calculate_view():
    return render_template('calculate.html')


@app.route('/calculate', methods=['POST'])
def calculate_post():
    text = request.form['text']
    out = communicate.compute(text)
    text = communicate.convert(text)

    print("[App]: Out [" + out + "]")
    if len(out) >= 26 and out[:5] == "GRAPH":
        results = parse("GRAPH[{}]: {}", out)

        date = results[1]
        ftr = results[0]

        print("[App]: Date is " + date)
        print("[App]: Ftr is " + ftr)

        return render_template('calculate.html',
                               input=text, out=out, date=date, ftr=ftr)

    print("[App]: In [" + text + "]")
    print("[App]: Out [" + out + "]")
    print("[App]: Alt-Path")

    return render_template('calculate.html', input=text, out=out)


logs = []


@app.route('/session')
def show_session_view():
    return render_template('session.html')


@app.route('/session', methods=['POST'])
def session_post():
    text = request.form['text']
    out = communicate.compute(text)
    text = communicate.convert(text)
    logs.insert(0, text + " = \\textbf{" + out + "}")
    return render_template('session.html', logs=logs)
