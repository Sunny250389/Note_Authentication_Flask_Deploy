# Flask Tutorial by Krish naik
from flask import Flask, request
import numpy as np
import pandas as pd
import pickle

# __name__ = from where the code is going to start
app = Flask(__name__)
pickle_in = open('classifier.pkl', 'rb')
classifier = pickle.load(pickle_in)


@app.route('/')
def welcome():
    return "Welcome to the app"


@app.route('/predict')
def predict_note_authentication():
    variance = request.args.get('variance')
    skewness = request.args.get('skewness')
    curtosis = request.args.get('curtosis')
    entropy = request.args.get('entropy')
    prediction = classifier.predict([[variance, skewness, curtosis, entropy]])
    return "The prediction value is" + str(prediction)


@app.route('/predict_file', methods=["POST"])
def predict_note_file():
    df_test = pd.read_csv(request.files.get("file"))
    prediction = classifier.predict(df_test)
    return "The prediction value for the file is" + str(list(prediction))



if __name__ == "__main__":
    app.run()
