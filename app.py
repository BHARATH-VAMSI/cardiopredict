from flask import Flask, request, render_template
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler


app = Flask(__name__)
model = pickle.load(open('lstmmodel.pkl', 'rb'))
sc = StandardScaler()


@app.route('/')
def hello():
    return render_template("index.html")


@app.route('/submit', methods = ["POST"])
def submit():
    age = request.form.get('age')
    gender = request.form.get('gender')
    height = request.form.get('height')
    weight = request.form.get('weight')
    ap_hi = request.form.get('ap_hi')
    ap_lo = request.form.get('ap_lo')
    cholestrol = request.form.get('cholestrol')
    glucose = request.form.get('glucose')
    smoke = request.form.get('smoke')
    alcohol = request.form.get('alcohol')
    active = request.form.get('active')
    user_input = np.array([[age, gender, height, weight, ap_hi, ap_lo, cholestrol, glucose, smoke, alcohol, active]])
    user_input = sc.transform(user_input)
    user_input = np.reshape(user_input, (user_input.shape[0], 1, user_input.shape[1]))
    prediction = model.predict(user_input)

    return render_template('index1.html', res=prediction)


if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0')