from flask import Flask, jsonify, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import tempfile

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

app = Flask(__name__)

os.makedirs('models', exist_ok=True)

models_loaded = True
try:
    if os.path.exists('models/pneumonia_model.h5'):
        cnn_model = load_model('models/pneumonia_model.h5')
        print(cnn_model.input_shape)
    else:
        print("Warning: models/pneumonia_model.h5 not found")
        models_loaded = False
        
    if os.path.exists('models/resnet50_pneumonia_finetuned.h5'):
        resnet_model = load_model('models/resnet50_pneumonia_finetuned.h5')
        print(resnet_model.input_shape)

    else:
        print("Warning: models/resnet50_pneumonia_finetuned.h5 not found")
        models_loaded = False
except Exception as e:
    print(f"Error loading models: {e}")
    models_loaded = False

def image_preprocess(img_path, target_size=(224, 224)):
    """Process an image for neural network prediction"""
    img = image.load_img(img_path, target_size=target_size)
    image_arr = image.img_to_array(img)
    image_arr = np.expand_dims(image_arr, axis=0)
    image_arr /= 255.0
    return image_arr


def image_preprocess_cnn(img_path, target_size=(128, 128)):
    """Process an image for neural network prediction"""
    img = image.load_img(img_path, target_size=target_size)
    image_arr = image.img_to_array(img)
    image_arr = np.expand_dims(image_arr, axis=0)
    image_arr /= 255.0
    return image_arr

@app.route('/')
def index():
    if models_loaded:
        return 'Pneumonia Detection API is running. Models loaded successfully.'
    else:
        return 'Pneumonia Detection API is running, but models failed to load. Check server logs.'

@app.route('/predict/resnet', methods=['POST'])
def resnet_predict():
    if not models_loaded or 'resnet_model' not in globals():
        return jsonify({"error": "ResNet model not loaded"}), 500
        
    if "image" not in request.files:
        return jsonify({"error": "No image file found in request"}), 400
    
    image_file = request.files["image"]
    
    # Save the uploaded file to a temporary location
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
    temp_path = temp_file.name
    temp_file.close()
    
    try:
        image_file.save(temp_path)
        
        img = image_preprocess(img_path=temp_path)
        
        result = resnet_model.predict(img)
        
        prediction = float(result[0][0])
        diagnosis = "Pneumonia" if prediction > 0.5 else "Normal"
        confidence = prediction if prediction > 0.5 else 1 - prediction
        
        return jsonify({
            "diagnosis": diagnosis,
            "confidence": float(confidence),
            "raw_prediction": float(prediction)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.route('/predict/cnn', methods=['POST'])
def cnn_predict():
    if not models_loaded or 'cnn_model' not in globals():
        return jsonify({"error": "CNN model not loaded"}), 500
        
    if "image" not in request.files:
        return jsonify({"error": "No image file found in request"}), 400
    
    image_file = request.files["image"]
    
    # Save the uploaded file to a temporary location
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
    temp_path = temp_file.name
    temp_file.close()
    
    try:
        # Save the uploaded file
        image_file.save(temp_path)
        
        # Preprocess the image
        img = image_preprocess_cnn(img_path=temp_path)
        
        # Make prediction
        result = cnn_model.predict(img)
        
        # Process prediction result (assuming binary classification)
        prediction = float(result[0][0])
        diagnosis = "Pneumonia" if prediction > 0.5 else "Normal"
        confidence = prediction if prediction > 0.5 else 1 - prediction
        
        return jsonify({
            "diagnosis": diagnosis,
            "confidence": float(confidence),
            "raw_prediction": float(prediction)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "models_loaded": models_loaded})

if __name__ == '__main__':
    app.run(debug=True)