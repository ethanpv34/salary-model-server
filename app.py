from flask import Flask, request, jsonify
import pickle
import numpy as np
from flask_cors import CORS, cross_origin
from utils.constants import client_url

app = Flask(__name__)
CORS(app, origins=client_url)

# Load in our model
def load_model():
  with open("saved_steps.pkl", "rb") as file:
    data = pickle.load(file)
  return data

data = load_model()

regressor = data["model"]
le_country = data["le_country"]
le_education = data["le_education"]

@app.route("/predict", methods=["POST"])
def predict_salary():
  if request.is_json:
    req_data = request.get_json()
    country = req_data.get("country")
    experience = req_data.get("experience")
    education = req_data.get("education")

    if not country or not experience or not education:
      return jsonify({"error": "Missing required fields"}), 400

    X = np.array([[country, education, experience]])
    X[:, 0] = le_country.transform(X[:,0])
    X[:, 1] = le_education.transform(X[:,1])
    X = X.astype(float)

    salary = regressor.predict(X)
    salary_list = salary.tolist()
    rounded_salary = round(salary_list[0])
    formatted_salary = "${:,.0f}".format(rounded_salary)

    return jsonify({"salary": formatted_salary})
  else:
    return jsonify({"error": "Request data expected in JSON"})
  
if __name__ == "__main__":
  app.run(debug=True)
