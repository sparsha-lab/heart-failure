from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load pre-trained model and label encoder
with open("heart.rf-model.pkl", "rb") as file:
    heart_model = pickle.load(file)

with open("lb-heart.pkl", "rb") as file:
    lb_heart = pickle.load(file)

# Prediction function
def predict_stroke(age=45, anaemia=0, creatinine_phosphokinase=56, diabetes=0, ejection_fraction=50, 
                   high_blood_pressure=0, platelets=263358.03, serum_creatinine=2, serum_sodium=120, 
                   sex=1, smoking=1, time=56):
    # Prepare the input list for prediction
    lst = [age, anaemia, creatinine_phosphokinase, diabetes, ejection_fraction, high_blood_pressure,
           platelets, serum_creatinine, serum_sodium, sex, smoking, time]
    
    # Make the prediction
    result = heart_model.predict([lst])
    
    # If lb_heart is a LabelEncoder, use inverse_transform, else handle result manually
    if hasattr(lb_heart, 'inverse_transform'):
        prediction = lb_heart.inverse_transform(result)[0]
    else:
        # Manually map the result if not a LabelEncoder
        prediction = "Heart Failure" if result == 1 else "No Heart Failure"
    
    return prediction


# Routes for the app
@app.route("/", methods=["GET"])
def index():
    return render_template("home.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        # Collect the form data from the user
        age = int(request.form.get("age"))
        anaemia = 1 if request.form.get("anaemia") == "Yes" else 0
        creatinine_phosphokinase = float(request.form.get("creatinine_phosphokinase"))
        diabetes = 1 if request.form.get("diabetes") == "Yes" else 0
        ejection_fraction = float(request.form.get("ejection_fraction"))
        high_blood_pressure = 1 if request.form.get("high_blood_pressure") == "Yes" else 0
        platelets = float(request.form.get("platelets"))
        serum_creatinine = float(request.form.get("serum_creatinine"))
        serum_sodium = float(request.form.get("serum_sodium"))
        sex = 1 if request.form.get("sex") == "Male" else 0
        smoking = 1 if request.form.get("smoking") == "Yes" else 0
        time = int(request.form.get("time"))

        # Make prediction
        prediction = predict_stroke(age, anaemia, creatinine_phosphokinase, diabetes, ejection_fraction,
                                    high_blood_pressure, platelets, serum_creatinine, serum_sodium, sex, smoking, time)

        return render_template("predict.html", prediction=prediction)

    return render_template("predict.html")


@app.route("/about", methods=["GET"])
def about():
    return render_template("about.html")


@app.route("/contact", methods=["GET"])
def contact():
    return render_template("contact.html")


if __name__ == "__main__":
    app.run(debug=True,host='0.0.0.0',port=8000)
