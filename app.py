from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load model + scaler saat server start
with open("model_air.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler_air.pkl", "rb") as f:
    scaler = pickle.load(f)

# Urutan fitur HARUS sama seperti training:
FEATURES = [
    "ph",
    "Hardness",
    "Solids",
    "Chloramines",
    "Sulfate",
    "Conductivity",
    "Organic_carbon",
    "Trihalomethanes",
    "Turbidity",
]

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", result=None)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Ambil input dari form sesuai urutan FEATURES
        values = []
        for f_name in FEATURES:
            raw = request.form.get(f_name, "").strip()
            if raw == "":
                return render_template("index.html", result=f"Kolom '{f_name}' wajib diisi.")
            values.append(float(raw))

        X = np.array(values).reshape(1, -1)
        X_scaled = scaler.transform(X)
        pred = model.predict(X_scaled)[0]

        if int(pred) == 1:
            msg = "✅ Air diprediksi LAYAK MINUM (Potable)"
        else:
            msg = "❌ Air diprediksi TIDAK LAYAK MINUM (Not Potable)"

        return render_template("index.html", result=msg)

    except Exception as e:
        return render_template("index.html", result=f"Error: {e}")

if __name__ == "__main__":
    app.run(debug=True)
