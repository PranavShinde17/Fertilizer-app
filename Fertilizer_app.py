from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

# Load the model and label encoder
model = pickle.load(open('classifier.pkl', 'rb'))
ferti_encoder = pickle.load(open('fertilizer.pkl', 'rb'))

@app.route('/')
def home():
    return "Welcome to the Fertilizer Prediction API!"

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from the request
    data = request.json  # Expect JSON input
    try:
        # Extract input features
        input_features = [data['Temperature'], data['Humidity'], data['Moisture'],
                          data['Soil_Type'], data['Crop_Type'], data['Nitrogen'],
                          data['Phosphorus'], data['Potassium']]
        # Predict the fertilizer class
        predicted_class = model.predict([input_features])[0]

        # Decode the fertilizer name
        fertilizer_name = ferti_encoder.classes_[predicted_class]

        return jsonify({
            'fertilizer': fertilizer_name,
            'message': 'Prediction successful'
        })
    except Exception as e:
        return jsonify({
            'error': str(e),
            'message': 'Prediction failed'
        }), 400

if __name__ == '__main__':
    app.run(debug=True)
