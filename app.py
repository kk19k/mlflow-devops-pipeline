from flask import Flask, request, jsonify
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from prometheus_client import Counter, generate_latest

app = Flask(__name__)

# Load registered MLflow model.
# If the MLflow registry is unavailable, train a fallback model
# so the Docker deployment remains self-contained.
try:
    model = mlflow.sklearn.load_model(
        "models:/IrisRandomForest/1"
    )
    print("Loaded model from MLflow Model Registry")
except Exception as e:
    print("MLflow model unavailable. Using fallback RandomForest model.")
    print(f"Reason: {e}")

    iris = load_iris()
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42
    )
    model.fit(iris.data, iris.target)

# Prometheus counter
prediction_counter = Counter(
    "prediction_requests_total",
    "Total prediction requests"
)

@app.route("/")
def home():
    return "ML Model API is running"

@app.route("/predict", methods=["POST"])
def predict():

    prediction_counter.inc()

    data = request.json["features"]

    prediction = model.predict([data])

    return jsonify({
        "prediction": int(prediction[0])
    })

@app.route("/metrics")
def metrics():
    return generate_latest(), 200, {
        "Content-Type": "text/plain; version=0.0.4; charset=utf-8"
    }

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
