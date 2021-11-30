
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle

from flask import Flask, request, render_template
from flask_sqlalchemy import SQLAlchemy


app = Flask(__name__)

coeur = pd.read_excel("./dataset/coeur.xlsx")
df = coeur.copy()

"""
(Test)

# Normalise les données.
for column in coeur.drop(["CŒUR"], axis=1).select_dtypes(np.number).columns:
    coeur[column] = coeur[column] / coeur[column].max()

# Encode les données.
for column in coeur.drop(["CŒUR"], axis=1).select_dtypes("object").columns:
    coeur[column] = coeur[column].astype("category").cat.codes

x = coeur.drop("CŒUR", axis=1)
y = coeur["CŒUR"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
"""


# Importe le modèle de régression logistique.
model = pickle.load(open("model.pkl", "rb"))

# Crée la base de données.
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///coeur-predict"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = True

db = SQLAlchemy(app)


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/form/", methods=["POST", "GET"])
def form():
    return render_template("simple_predict.html")


@app.route("/simple_predict/", methods=["POST", "GET"])
def simple_predict():
    # Enregistre les données du formulaire.
    data = {
        "AGE": int(request.form["age"]),
        "SEXE": request.form["sexe"],
        "TDT": request.form["tdt"],
        "PAR": int(request.form["par"]),
        "CHOLESTEROL": int(request.form["cholesterol"]), 
        "GAJ": int(str(request.form["gaj"])),
        "ECG": request.form["ecg"],
        "FCMAX": int(request.form["fcmax"]),
        "ANGINE": request.form["angine"],
        "DEPRESSION ": float(request.form["depression"]),
        "PENTE": request.form["pente"]
    }

    input_values = pd.DataFrame(data, index=[0])

    # Normalise les données.
    for column in df.drop(["CŒUR"], axis=1).select_dtypes(np.number).columns:
        input_values[column] = input_values[column] / df[column].max()

    # Encode les données.
    for column in df.drop(["CŒUR"], axis=1).select_dtypes("object").columns:
        input_values[column] = input_values[column].astype("category").cat.codes

    # Le modèle effectue la prédiction.
    prediction = model.predict(input_values)

    if prediction[0] == 0:
        retour = "Coeur sain"
    else:
        retour = "Coeur malade"

    return render_template("simple_predict.html", retour=f"{retour}")


@app.route("/multiple_predict/", methods=["POST", "GET"])
def multiple_predict():
    """
    (Test)

    # Error.
    value = request.form["individu"]
    liste = []

    for elt in range(1, int(value) + 1):
        tests = x_test.iloc[elt, :].ravel()
        tests = tests.reshape(1, tests.shape[0])
        prediction = model.predict(tests)

        if prediction[0] == 0:
            result = f"Individu {elt} : Coeur sain"
        else:
            result = f"Individu {elt} : Coeur malade"

        liste.append(result)
    """

    return render_template("multiple_predict.html")


if __name__ == "__main__":
    # db.create_all()
    app.run(debug=True)
