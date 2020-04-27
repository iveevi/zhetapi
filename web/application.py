import communicate

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
    out = communicate.compute(text)
    text = communicate.convert(text)
    return render_template('calculate.html', input = text, out = out)

logs = []

@app.route('/session')
def show_session_view():
    return render_template('session.html')

@app.route('/session', methods=['POST'])
def session_post():
    text = request.form['text']
    out = communicate.compute(text)
    text = communicate.convert(text)
    logs.insert(0, text + " = \\textbf{" + out + "}");
    return render_template('session.html', logs = logs)
