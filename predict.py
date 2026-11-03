from flask import Flask, request, render_template
import pickle
import xgboost as xgb
import numpy as np

app = Flask(__name__)

print("Loading model and vectorizer...")
model = xgb.Booster()
model.load_model('model.json')

with open('dv.pkl', 'rb') as f:
    dv = pickle.load(f)

feature_names = list(dv.get_feature_names_out())

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        form_data = request.form
        
        input_dict = {
            'season': str(form_data['season']),
            'yr': str(form_data['yr']),
            'mnth': str(form_data['mnth']),
            'holiday': str(form_data['holiday']),
            'weekday': str(form_data['weekday']),
            'workingday': str(form_data['workingday']),
            'weathersit': str(form_data['weathersit']),
            'temp': float(form_data['temp']),
            'hum': float(form_data['hum']),
            'windspeed': float(form_data['windspeed'])
        }
        
        hr = float(form_data['hr'])
        input_dict['hr_sin'] = np.sin(2 * np.pi * hr / 24)
        input_dict['hr_cos'] = np.cos(2 * np.pi * hr / 24)
        
        X_input = dv.transform([input_dict])
        dinput = xgb.DMatrix(X_input, feature_names=feature_names)
        
        prediction = model.predict(dinput)[0]
        
        result_text = f"Predicted Bike Rentals: {prediction:.0f} bikes"
        return render_template('index.html', prediction_text=result_text)
        
    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)