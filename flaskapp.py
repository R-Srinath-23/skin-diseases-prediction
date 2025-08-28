# app.py
from flask import Flask, request, jsonify
import os
from skin_prediction.prediction import predict_image
from flask_cors import CORS


app = Flask(__name__)
CORS(app)


UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Skin Disease Classifier API is running."})

@app.route("/predict", methods=["POST"])
def predict():
    print('request.files > ', request.files)
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    

    file = request.files['image']
    # print('datatype ----- ',type(file))

    

    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    image_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(image_path)

    print("Image saved to:", image_path)

    

    try:
        prediction = predict_image(image_path)
        os.remove(image_path)  # optional: remove uploaded file
        print('datatype',type(file))
        return jsonify({"prediction": prediction})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print('Running as main')
    app.run(debug=True)