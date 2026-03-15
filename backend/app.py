from flask import Flask, request, jsonify
from flask_cors import CORS
from predictor import predict_email

app = Flask(__name__)
CORS(app)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data or "email" not in data:
        return jsonify({"error": "No email provided"}), 400

    email_text = data["email"]
    if not email_text.strip():
        return jsonify({"error": "Email is empty"}), 400

    label, confidence = predict_email(email_text)
    return jsonify({
        "label": label,
        "confidence": confidence
    })

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(debug=True, port=5000)