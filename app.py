from flask import Flask, jsonify, request
from tensorflow.keras.preprocessing import image
import numpy as np
import tensorflow as tf
import os
import tempfile

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

app = Flask(__name__)
os.makedirs('models', exist_ok=True)

# Helper function to load TFLite models
def load_tflite_model(path):
    if os.path.exists(path):
        interpreter = tf.lite.Interpreter(model_path=path)
        interpreter.allocate_tensors()
        return interpreter
    else:
        print(f"Warning: {path} not found")
        return None

# Load models
cnn_model = load_tflite_model('models/pneumonia_model.tflite')
resnet_model = load_tflite_model('models/resnet50_pneumonia_finetuned.tflite')
models_loaded = cnn_model is not None and resnet_model is not None

# Prediction with TFLite model
def tflite_predict(interpreter, input_data):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], input_data.astype(np.float32))
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    return output_data

# Preprocess functions
def image_preprocess(img_path, target_size=(224, 224)):
    img = image.load_img(img_path, target_size=target_size)
    image_arr = image.img_to_array(img)
    image_arr = np.expand_dims(image_arr, axis=0)
    image_arr /= 255.0
    return image_arr

def image_preprocess_cnn(img_path, target_size=(128, 128)):
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
    if not models_loaded or resnet_model is None:
        return jsonify({"error": "ResNet model not loaded"}), 500

    if "image" not in request.files:
        return jsonify({"error": "No image file found in request"}), 400

    image_file = request.files["image"]
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
    temp_path = temp_file.name
    temp_file.close()

    try:
        image_file.save(temp_path)
        img = image_preprocess(temp_path)
        result = tflite_predict(resnet_model, img)
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
    if not models_loaded or cnn_model is None:
        return jsonify({"error": "CNN model not loaded"}), 500

    if "image" not in request.files:
        return jsonify({"error": "No image file found in request"}), 400

    image_file = request.files["image"]
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
    temp_path = temp_file.name
    temp_file.close()

    try:
        image_file.save(temp_path)
        img = image_preprocess_cnn(temp_path)
        result = tflite_predict(cnn_model, img)
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

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "models_loaded": models_loaded})

if __name__ == '__main__':
    app.run(debug=True)
