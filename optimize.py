import tensorflow as tf

model_cnn = tf.keras.models.load_model('models/pneumonia_model.h5')
model_resnet = tf.keras.models.load_model('models/resnet50_pneumonia_finetuned.h5')

converter = tf.lite.TFLiteConverter.from_keras_model(model_resnet)
tflite_model = converter.convert()

with open('models/resnet50_pneumonia_finetuned.tflite', 'wb') as f:
    f.write(tflite_model)


print("Model converted to TFLite and saved.")
