# importing required libraries

import warnings

import joblib
import numpy as np
from flask import Flask, render_template, request,redirect

warnings.filterwarnings("ignore")
from feature import FeatureExtraction

gbc = joblib.load("model.pkl")


app = Flask(__name__)

@app.route("/")
def homepage():
    return render_template("index.html")

@app.route("/signin")
def signin():
    return render_template("sign.html")

@app.route("/model", methods=["GET", "POST"])
def model():
    if request.method == "POST":
        url = request.form["url"]
        obj = FeatureExtraction(url)
        x = np.array(obj.getFeaturesList()).reshape(1, 30)

        y_pred = gbc.predict(x)[0]
        # 1 is safe
        # -1 is unsafe
        y_pro_phishing = gbc.predict_proba(x)[0, 0]
        y_pro_non_phishing = gbc.predict_proba(x)[0, 1]
        # if(y_pred ==1 ):
        pred = "It is {0:.2f} % safe to go ".format(y_pro_phishing * 100)
        return render_template("main.html", xx=round(y_pro_non_phishing, 2), url=url)
    return render_template("main.html", xx=-1)


if __name__ == "__main__":
    app.run(debug=True)
