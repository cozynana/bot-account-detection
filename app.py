from flask import Flask, request, jsonify,  render_template
from flask_cors import CORS
import joblib
import pandas as pd

app = Flask(__name__)
CORS(app)

# Load model dan scaler
model = joblib.load('model_v2.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def index():
    return render_template('index2.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    # Ambil fitur input dari user
    features = data['features']  # [followers, following, tweets, accountAge, ratio, tweetsPerYear]

    # Definisi nama kolom yang sama seperti saat scaler dilatih
    feature_names = ['Jumlah Follower', 'Jumlah Following', 'Jumlah Tweet', 'Umur Akun (tahun)', 'Rasio_Follower_Following', 'Tweet_Per_Tahun']

    # Konversi input menjadi DataFrame dengan nama kolom
    input_df = pd.DataFrame([features], columns=feature_names)

    # Lakukan scaling pada fitur input
    input_df[feature_names] = scaler.transform(input_df)

    # Prediksi menggunakan model
    prediction = model.predict(input_df)[0]

    # Konversi hasil prediksi ke label
    prediction_label = "akun bot" if prediction == 'bot' else "bukan akun bot"
    return jsonify({'prediction': prediction_label})

if __name__ == '__main__':
    app.run(debug=True)