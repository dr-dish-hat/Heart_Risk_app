from flask import Flask, render_template, request
import pickle
import numpy as np

model = pickle.load(open('model.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def man():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def home():
    data1 = request.form['a']
    data2 = request.form['b']
    data3 = request.form['c']
    data4 = request.form['d']
    data5 = request.form['e']
    data6 = request.form['f']

    data = request.form['g']
    if data == 'yes' or 'Yes' or 'YES':
        data7 = 1
    else:
        data7 = 0

    data = request.form['h']
    if data == 'yes' or 'Yes' or 'YES':
        data8 = 1
    else:
        data8 = 0

    arr = np.array([[data1, data2, data3, data4, data5, data6, data7, data8]])
    pred = model.predict(arr)

    return render_template('predict.html', data = pred)


if __name__ == "__main__":
    app.run(debug=True)
