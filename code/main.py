import pickle

from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

model = joblib.load("model.pkl")



crop_images = {
    "rice": "rice.jpg",
    "cotton": "cotton.jpg",
    "maize": "maize.jpg",
    "banana": "banana.jpg",
    "mango": "mango.jpg"
}

crop_telugu = {
    "rice": "వరి",
    "cotton": "పత్తి",
    "maize": "మొక్కజొన్న",
    "banana": "అరటి",
    "mango": "మామిడి"
}


@app.route("/")
def home():
    return render_template("home.html")

@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/findyourcrop")
def findyourcrop():
    return render_template("findyourcrop.html")

@app.route("/confusionmatrix")
def confusionmatrix():
    return render_template("confusion_matrix.html")


@app.route("/modelcomparison")
def modelcomparison():
    return render_template("model_comparison.html")


@app.route("/predict", methods=["POST"])
def predict():

    features = [
        float(request.form["Nitrogen"]),
        float(request.form["Phosphorus"]),
        float(request.form["Potassium"]),
        float(request.form["Temperature"]),
        float(request.form["Humidity"]),
        float(request.form["pH"]),
        float(request.form["Rainfall"])
    ]

    prediction = model.predict([features])[0]

    crop = str(prediction).lower()

    image = crop_images.get(crop, "farm.jpg")

    telugu_crop = crop_telugu.get(crop, crop)

    return render_template(
        "result.html",
        crop=prediction,
        image=image,
        telugu_crop=telugu_crop
    )


if __name__ == "__main__":
    app.run(debug=True)
    